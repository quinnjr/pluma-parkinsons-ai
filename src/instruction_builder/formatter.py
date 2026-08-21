"""Turn Stage-1 model output into grounded instruction-response pairs.

Every sentence in a generated response traces to one of exactly three sources:

* the pipeline's own numbers (SHAP attributions, MOFA+ factors, z-scores),
* a curated, PubMed-verified claim from :mod:`src.knowledge`,
* an explicit statement that no curated evidence exists.

There is no fourth category. The previous version of this module appended
"This feature is a known contributor to PD pathophysiology" to every biomarker,
which is a fabrication for any feature outside the curated set and was false
even for real genes in healthy-control profiles.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.instruction_builder.templates import SYSTEM_PROMPT, TASK_INSTRUCTIONS
from src.knowledge import Entity, KnowledgeBase, load_knowledge_base
from src.models import Stage1Output

#: Concordance verdicts between an observed value and the curated PD direction.
CONCORDANT = "concordant"
DISCORDANT = "discordant"
UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class GroundedHit:
    """A biomarker paired with whatever the knowledge base does or does not know."""

    modality: str
    feature: str
    shap_value: float
    pushes_toward_pd: bool
    value_z: float | None
    entity: Entity | None

    @property
    def is_annotated(self) -> bool:
        return self.entity is not None and self.entity.has_pd_claim

    @property
    def concordance(self) -> str:
        """Does the subject's observed value move the way the literature reports?"""
        if not self.is_annotated or self.value_z is None:
            return UNDETERMINED
        expected = self.entity.pd_association.direction
        if expected not in {"up", "down"}:
            return UNDETERMINED
        observed = "up" if self.value_z > 0 else "down"
        return CONCORDANT if observed == expected else DISCORDANT

    @property
    def citation_keys(self) -> tuple[str, ...]:
        if not self.is_annotated:
            return ()
        return tuple(c.key for c in self.entity.pd_association.citations)

    @property
    def pmids(self) -> tuple[str, ...]:
        if not self.is_annotated:
            return ()
        return tuple(c.pmid for c in self.entity.pd_association.citations)


def _push(hit: GroundedHit) -> str:
    return "toward" if hit.pushes_toward_pd else "away from"


def _value_phrase(hit: GroundedHit) -> str:
    if hit.value_z is None:
        return "the subject's value was not standardised against the cohort"
    level = "elevated" if hit.value_z > 0 else "reduced"
    return f"the subject's value is {level} relative to the cohort (z = {hit.value_z:+.2f})"


class InstructionFormatter:
    """Build instruction-response pairs grounded in the curated knowledge base."""

    def __init__(self, seed: int = 42, knowledge_base: KnowledgeBase | None = None,
                 top_n: int = 5):
        self._rng = random.Random(seed)
        self.kb = knowledge_base or load_knowledge_base()
        self.top_n = top_n

    # -- grounding -------------------------------------------------------------

    def ground(self, output: Stage1Output, limit: int | None = None) -> list[GroundedHit]:
        hits = output.top_biomarkers[: limit or self.top_n]
        return [
            GroundedHit(
                modality=hit.modality,
                feature=hit.feature,
                shap_value=hit.shap_value,
                pushes_toward_pd=hit.pushes_toward_pd,
                value_z=hit.value_z,
                entity=self.kb.lookup(hit.feature, hit.modality),
            )
            for hit in hits
        ]

    # -- shared blocks ---------------------------------------------------------

    def _format_profile(self, output: Stage1Output) -> str:
        lines = [
            f"Subject: {output.subject_id}",
            f"Recorded diagnosis: {output.diagnosis}",
            f"Environmental risk score: {output.environmental_risk_score:.1f}/10",
            "",
            "Top biomarkers by |SHAP| (XGBoost + TreeSHAP):",
        ]
        for hit in self.ground(output, limit=len(output.top_biomarkers)):
            z = "" if hit.value_z is None else f", z={hit.value_z:+.2f}"
            lines.append(
                f"  - [{hit.modality}] {hit.feature}: SHAP={hit.shap_value:+.3f}"
                f" ({_push(hit)} PD){z}"
            )
        if output.mofa_factors:
            top_factors = sorted(
                output.mofa_factors.items(), key=lambda kv: abs(kv[1]), reverse=True
            )[:3]
            lines.append("")
            lines.append("MOFA+ latent factors (top 3 by magnitude):")
            lines.extend(f"  - {name}: {value:+.3f}" for name, value in top_factors)

        prov = output.provenance
        lines.append("")
        lines.append("Provenance:")
        lines.append(f"  - cohort size: {prov.cohort_size or 'unreported'}")
        lines.append(
            f"  - SHAP attributions: {'out-of-fold' if prov.shap_out_of_fold else 'in-sample'}"
        )
        if prov.datasets:
            lines.append(f"  - datasets: {', '.join(prov.datasets)}")
        if prov.synthetic_modalities:
            lines.append(f"  - simulated modalities: {', '.join(prov.synthetic_modalities)}")
        if prov.cv_auc is not None:
            lines.append(f"  - cross-validated ROC AUC: {prov.cv_auc:.2f}")
        return "\n".join(lines)

    def _caveats(self, output: Stage1Output, grounded: list[GroundedHit]) -> list[str]:
        """Limitations that are true of this specific subject's evidence."""
        prov = output.provenance
        caveats = []
        if prov.cohort_size and prov.cohort_size < 100:
            caveats.append(
                f"The cohort has {prov.cohort_size} subjects. Feature rankings from a "
                f"cohort this small are unstable and should be treated as hypothesis-generating."
            )
        if not prov.shap_out_of_fold:
            caveats.append(
                "SHAP attributions were computed in-sample, on the same subjects the "
                "classifier was fitted to, which inflates apparent feature importance."
            )
        if prov.cv_auc is not None and prov.cv_auc < 0.7:
            caveats.append(
                f"The classifier's cross-validated ROC AUC is {prov.cv_auc:.2f}, so its "
                f"per-subject attributions rest on weak discrimination."
            )
        simulated = [m for m in prov.synthetic_modalities if m in {h.modality for h in grounded}]
        if simulated:
            caveats.append(
                f"Values for the following modalities were simulated, not measured: "
                f"{', '.join(sorted(set(simulated)))}. No biological conclusion about this "
                f"subject can rest on them."
            )
        unannotated = [h for h in grounded if not h.is_annotated]
        if unannotated:
            caveats.append(
                f"{len(unannotated)} of the {len(grounded)} features listed have no curated "
                f"PD annotation and are reported as statistical associations only."
            )
        return caveats

    def _references(self, grounded: list[GroundedHit]) -> list[str]:
        seen: dict[str, str] = {}
        for hit in grounded:
            if not hit.is_annotated:
                continue
            for citation in hit.entity.pd_association.citations:
                seen.setdefault(citation.pmid, citation.render())
        return [seen[pmid] for pmid in sorted(seen, key=lambda p: int(p))]

    def _render(self, body: list[str], caveats: list[str], references: list[str]) -> str:
        sections = ["\n".join(body).rstrip()]
        if caveats:
            sections.append("Limitations:\n" + "\n".join(f"- {c}" for c in caveats))
        if references:
            sections.append("References:\n" + "\n".join(f"- {r}" for r in references))
        else:
            sections.append(
                "References: none. No feature in this profile carries a curated, "
                "citation-backed PD association."
            )
        return "\n\n".join(sections)

    def _pair(self, task: str, output: Stage1Output, grounded: list[GroundedHit],
              response: str) -> dict:
        return {
            "task": task,
            "subject_id": output.subject_id,
            "instruction": self._rng.choice(TASK_INSTRUCTIONS[task]),
            "input": self._format_profile(output),
            "output": response,
            "grounding": {
                "pmids": sorted({p for hit in grounded for p in hit.pmids}, key=int),
                "annotated_features": [h.feature for h in grounded if h.is_annotated],
                "unannotated_features": [h.feature for h in grounded if not h.is_annotated],
                "synthetic_modalities": list(output.provenance.synthetic_modalities),
            },
        }

    # -- tasks -----------------------------------------------------------------

    def biomarker_discovery(self, output: Stage1Output) -> dict:
        grounded = self.ground(output)
        body = [
            f"Ranked biomarkers for subject {output.subject_id}, as attributed by the "
            f"classifier (these are model attributions, not measured effect sizes):",
            "",
        ]
        for i, hit in enumerate(grounded, 1):
            body.append(
                f"{i}. {hit.feature} [{hit.modality}] — SHAP {hit.shap_value:+.3f}, pushing "
                f"{_push(hit)} a PD classification; {_value_phrase(hit)}."
            )
            if hit.entity is None:
                body.append(
                    "   No curated annotation for this feature. It is a statistical "
                    "association in this cohort; no biological claim is made."
                )
            else:
                body.append(f"   Function: {hit.entity.function}")
                if hit.is_annotated:
                    assoc = hit.entity.pd_association
                    body.append(f"   Reported in PD: {assoc.statement} [{assoc.render_citations()}]")
                    body.append(f"   Concordance: {self._concordance_sentence(hit)}")
                else:
                    body.append(
                        "   No curated PD association for this entity, so its appearance "
                        "here is a statistical association only."
                    )
            body.append("")
        return self._pair("biomarker_discovery", output, grounded,
                          self._render(body, self._caveats(output, grounded),
                                       self._references(grounded)))

    def _concordance_sentence(self, hit: GroundedHit) -> str:
        expected = hit.entity.pd_association.direction
        verdict = hit.concordance
        if verdict == CONCORDANT:
            return (
                f"the observed direction matches the reported {expected}-regulation in PD, "
                f"so the model's use of this feature is consistent with the literature."
            )
        if verdict == DISCORDANT:
            return (
                f"the observed direction is opposite to the reported {expected}-regulation "
                f"in PD. The classifier still weights it, so treat this as a cohort-specific "
                f"statistical signal rather than confirmation of the published effect."
            )
        if hit.value_z is None:
            return "no standardised value is available, so concordance cannot be assessed."
        return (
            f"the reported direction in PD is '{expected}', which does not define a single "
            f"expected sign, so concordance cannot be assessed."
        )

    def clinical_prediction(self, output: Stage1Output) -> dict:
        grounded = self.ground(output, limit=3)
        confidence = output.prediction_confidence * 100
        stage = f", staged {output.disease_stage}" if output.disease_stage else ""
        supporting = [h for h in grounded if h.pushes_toward_pd]
        opposing = [h for h in grounded if not h.pushes_toward_pd]

        body = [
            f"Model output: {output.diagnosis}{stage}, at {confidence:.1f}% predicted "
            f"probability of PD. This is a classifier output on research data, not a "
            f"clinical diagnosis.",
            "",
        ]
        if supporting:
            body.append("Features pushing toward a PD classification:")
            body.extend(f"- {self._evidence_line(h)}" for h in supporting)
            body.append("")
        if opposing:
            body.append("Features pushing away from a PD classification:")
            body.extend(f"- {self._evidence_line(h)}" for h in opposing)
            body.append("")

        annotated = [h for h in grounded if h.is_annotated]
        concordant = [h for h in annotated if h.concordance == CONCORDANT]
        body.append(
            f"Evidential weight: {len(annotated)} of the {len(grounded)} top features carry a "
            f"curated PD association, and {len(concordant)} of those move in the direction "
            f"the literature reports. "
            + (
                "The prediction is therefore supported by more than the classifier's internal "
                "geometry."
                if concordant
                else "The prediction therefore rests mainly on the classifier's internal "
                "geometry rather than on independently reported effects."
            )
        )
        body.append("")
        body.append(
            "To raise confidence: replicate in an independent cohort, confirm the "
            "attributions out of fold, and measure any simulated modality directly."
        )
        return self._pair("clinical_prediction", output, grounded,
                          self._render(body, self._caveats(output, grounded),
                                       self._references(grounded)))

    def _evidence_line(self, hit: GroundedHit) -> str:
        base = f"{hit.feature} [{hit.modality}], SHAP {hit.shap_value:+.3f}; {_value_phrase(hit)}"
        if hit.is_annotated:
            return f"{base}. {hit.entity.pd_association.statement} [{hit.entity.pd_association.render_citations()}]"
        return f"{base}. No curated PD annotation; statistical association only."

    def cross_modal_synthesis(self, output: Stage1Output) -> dict:
        grounded = self.ground(output)
        modalities = sorted({h.modality for h in grounded})
        annotated = [h for h in grounded if h.is_annotated]

        body = [
            f"Subject {output.subject_id} has top-ranked features spanning "
            f"{len(modalities)} modalities ({', '.join(modalities)}).",
            "",
        ]
        if len(annotated) >= 2:
            first, second = annotated[0], annotated[1]
            agree = first.pushes_toward_pd == second.pushes_toward_pd
            body.append(
                f"The two highest-ranked annotated features are {first.feature} "
                f"[{first.modality}] and {second.feature} [{second.modality}]. They push "
                + (
                    "the classification in the same direction"
                    if agree
                    else "the classification in opposite directions"
                )
                + f" ({_push(first)} PD and {_push(second)} PD respectively)."
            )
            if agree:
                body.append(
                    "Converging attributions across modalities are harder to explain by a "
                    "single measurement artefact than a signal confined to one assay, though "
                    "they can still share an upstream confounder such as medication or age."
                )
            else:
                body.append(
                    "The modalities disagree here. A cross-modal narrative that ignores the "
                    "opposing feature would misrepresent the evidence."
                )
            body.append("")
            body.append("What the literature supports for each:")
            for hit in annotated:
                body.append(
                    f"- {hit.feature} [{hit.modality}]: {hit.entity.pd_association.statement} "
                    f"[{hit.entity.pd_association.render_citations()}] "
                    f"Observed here: {self._concordance_sentence(hit)}"
                )
        elif len(annotated) == 1:
            hit = annotated[0]
            body.append(
                f"Only one top-ranked feature carries a curated PD association: "
                f"{hit.feature} [{hit.modality}]. {hit.entity.pd_association.statement} "
                f"[{hit.entity.pd_association.render_citations()}] A single annotated feature "
                f"does not establish a cross-modal pattern."
            )
        else:
            body.append(
                "None of the top-ranked features carries a curated PD association, so no "
                "cross-modal biological synthesis can be offered for this subject. The "
                "convergence visible here is between model attributions, not between "
                "independently established effects."
            )
        body.append("")
        unannotated = [h for h in grounded if not h.is_annotated]
        if unannotated:
            body.append(
                "Excluded from the synthesis for lack of curated evidence: "
                + ", ".join(f"{h.feature} [{h.modality}]" for h in unannotated)
                + "."
            )
        return self._pair("cross_modal_synthesis", output, grounded,
                          self._render(body, self._caveats(output, grounded),
                                       self._references(grounded)))

    def all_formats(self, output: Stage1Output) -> list[dict]:
        return [
            self.biomarker_discovery(output),
            self.clinical_prediction(output),
            self.cross_modal_synthesis(output),
        ]

    @staticmethod
    def system_prompt() -> str:
        return SYSTEM_PROMPT
