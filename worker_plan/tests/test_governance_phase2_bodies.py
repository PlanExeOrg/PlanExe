from worker_plan_internal.governance.governance_phase2_bodies import DocumentDetails


def test_document_details_coerces_list_fields_to_string():
    payload = {
        "internal_governance_bodies": [
            {
                "name": "Board",
                "rationale_for_inclusion": "Because.",
                "responsibilities": ["r1"],
                "initial_setup_actions": ["a1"],
                "membership": ["m1"],
                "decision_rights": ["line1", "line2"],
                "decision_mechanism": ["mech1", "mech2"],
                "meeting_cadence": ["Weekly"],
                "typical_agenda_items": ["item1"],
                "escalation_path": ["path1", "path2"],
            }
        ]
    }

    doc = DocumentDetails.model_validate(payload)
    body = doc.internal_governance_bodies[0]

    assert isinstance(body.decision_rights, str) and "line1" in body.decision_rights
    assert isinstance(body.decision_mechanism, str)
    assert isinstance(body.meeting_cadence, str)
    assert isinstance(body.escalation_path, str)
