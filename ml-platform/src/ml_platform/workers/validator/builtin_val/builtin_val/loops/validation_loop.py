import importlib
import torch
from torch.utils.data import DataLoader

import numpy as np
from ml_platform.workers.validator.plugin_api.validation_loop import ValidationLoop, ValidationResult
from ml_platform.workers.validator.plugin_api.context import ValidationContext
from ml_platform.workers.validator.validation_config import ValidationRule, ValidationType, Validation


class PyTorchClassifierValidationLoop(ValidationLoop):
    def __init__(self):
        self.sk_metrics = importlib.import_module("sklearn.metrics")
        self.dataset = None
        self.data_location = None

    def set_data_source(self):
        pass

    def _validate_rule( self, rule: ValidationRule, ctx: ValidationContext, model=None, ) -> tuple[float | None, bool]:

        result = False
        value = None

        data_source = ctx._get_data_source(rule.data_source)
        self.data_location = rule.data_source

        try:
            metric = getattr(self.sk_metrics, rule.metric)
        except AttributeError:
            print(
                f"Skipping metric {rule.rule_name}, "
                f"not in {self.available_metrics.keys()}"
            )
            return result, value

        target_key = ctx.validation_config.dataset_description.target_column[0]
        sample_key = ctx.validation_config.dataset_description.sample_column[0]
        device = ctx.validation_config.device

        # Get streaming dataset
        dataset = data_source.get_torch_dataset( split="val", num_splits=3, seed=0, max_samples=rule.n_samples)

        loader = DataLoader( dataset, batch_size=64, )

        model = ctx.model if model is None else model

        all_predictions = []
        all_targets = []
        all_probas = []

        try:
            with torch.no_grad():
                for batch in loader:

                    X = batch[sample_key].to(device)
                    y_true = batch[target_key].to(device)

                    out = model(X)
                    logits = out["logits"]
                    probas = torch.softmax(logits, dim=-1)
                    y_pred = torch.argmax(logits, dim=-1)

                    all_predictions.append(y_pred.cpu())
                    all_targets.append(y_true.cpu())
                    all_probas.append(probas.cpu())

            if not all_predictions:
                print(f"No samples available for rule {rule.rule_name}")
                return result, value

            y_pred = torch.cat(all_predictions).numpy()
            y_true = torch.cat(all_targets).numpy()
            probas = torch.cat(all_probas).numpy()

            value = metric(y_pred, y_true)

            if rule.min_value is not None and rule.max_value is not None:
                result = rule.min_value <= value <= rule.max_value
            elif rule.min_value is not None:
                result = value >= rule.min_value
            elif rule.max_value is not None:
                result = value <= rule.max_value
            else:
                result = True

        except Exception as e:
            print(f"Rule {rule.rule_name} failed with: {e}")

        return result, value


    def _validate_standalone(self, rules: list[ValidationRule], ctx) -> dict:
        results = []
        values = []
        for rule in rules:
            r, v = self._validate_rule(rule, ctx)
            results.append(r); values.append(v)
        
        return {
            "results": results,
            "values": values,
            "rules": rules
        }

    def _validate_ab(self, rules: list[ValidationRule], ctx) -> dict:
        results = []
        values = []

        model_a = ctx.model

        for rule in rules:
            model_b = ctx.mlf_logger.get_model(rule.model_b)

            ra, va = self._validate_rule(rule, ctx, model_a)
            rb, vb = self._validate_rule(rule, ctx, model_b)

            if not (ra and rb) or vb == 0:
                results.append(False)
                values.append(None)
                continue

            if rule.max_relative_change: # relative, TODO: make the check more explicit
                relative_change = (va - vb) / vb
                results.append(rule.min_relative_change <= relative_change < rule.max_relative_change)
                values.append(relative_change)

            elif rule.max_absolute_change: # aboslute
                absolute_change = va - vb
                results.append(rule.min_absolute_change <= absolute_change < rule.max_absolute_change)
                values.append(absolute_change)

        return {
            "results": results,
            "values": values,
            "rules": rules,
        }

    def validate(self, ctx: ValidationContext) -> ValidationResult:
        
        print("Validating")
        results = {}
        passed = True
        for validation in ctx.validation_config.validations:
            summary = self._validate(validation, ctx)
            for rule, result, value in zip(summary["rules"], summary["results"], summary["values"]):
                if not result:
                    print( f"Rule {rule.rule_name!s} failed with value {value}")
                    passed = False
                else:
                    print( f"Rule {rule.rule_name!s} passed with value {value}")

                key = f"{validation.validation_type!s}_{rule.rule_name}"
                ctx.mlf_logger.log_metric(key, value)
                results[key] = [result, value]
        
        return ValidationResult(
            validation_info={
                "passed": passed,
                "validation_results": results,
                "validation_config": ctx.validation_config
                }
            )


    def _validate(self, validation: Validation, ctx: ValidationContext):

        match validation.validation_type:
                    
            case ValidationType.STANDALONE:
                summary = self._validate_standalone(validation.rules, ctx)
            
            case ValidationType.AB_TEST:
                summary = self._validate_ab(validation.rules, ctx)

            case ValidationType.SHADOW:
                summary = self._validate_shadow()

            case _:
                raise TypeError(f"Invalid validation type '{self.validation_type}', should be one of {self.available_metrics}")
        
        return summary
