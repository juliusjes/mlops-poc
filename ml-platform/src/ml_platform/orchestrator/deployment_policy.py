def should_deploy(deploy_criteria: str, validation_result: dict) -> bool:
    """Apply the documented deployment criterion for a completed training event."""
    if deploy_criteria == "none":
        return True
    if deploy_criteria == "validated":
        return bool(validation_result.get("passed"))
    if deploy_criteria == "no_deployment":
        return False
    raise ValueError(f"Unsupported deploy criterion: {deploy_criteria}")
