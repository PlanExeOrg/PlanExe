from worker_plan_internal.expert.pre_project_assessment import ExpertDetails


def make_feedback_item(idx=1):
    return {
        "feedback_index": idx,
        "feedback_title": "Ensure test coverage",
        "feedback_description": "You must cover this area."
    }


def test_expert_details_accepts_missing_feedback_item_list():
    payload = {
        "expert1": {
            "expert_title": "Efficiency Lead",
            "expert_full_name": "Jane Doe",
            "feedback_item_list": [make_feedback_item(1)],
        },
        "expert2": {
            "expert_title": "Safety Lead",
            "expert_full_name": "John Smith",
            # feedback_item_list omitted on purpose
        },
        "combined_summary": "Three critical blockers remain.",
        "go_no_go_recommendation": "Proceed with Caution",
    }

    details = ExpertDetails.model_validate(payload)

    assert details.expert1.feedback_item_list
    assert details.expert2.feedback_item_list == []
    assert details.combined_summary.startswith("Three")
