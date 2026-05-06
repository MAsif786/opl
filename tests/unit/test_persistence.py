"""
Tests for WorldModel persistence (save/load).
"""

import os
import pytest
import numpy as np
from opl.model.world_model import WorldModel
from opl.model.rules import simple_stock_rule
from opl.state.vector import StateVector
from opl.model.action import Action

class TestPersistence:

    @pytest.mark.unit
    def test_save_and_load_correction_model(self, tmp_path):
        """Verify that a trained model can be saved and reloaded with identical predictions."""
        model = WorldModel(rule=simple_stock_rule)
        
        # 1. Record enough observations to train
        for i in range(15):
            s_t = StateVector([100, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])
            action = Action.no_op()
            # Systematic error: real stock is always 5 units lower than rule predicts (e.g. theft/spoilage)
            # Rule predicts 100-20=80. Real is 75.
            s_real = StateVector([75, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])
            model.record_observation(s_t, action, s_real)
            
        model.train_correction()
        
        # Check prediction before saving
        s_test = StateVector([200, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])
        pred_before = model.predict(s_test, Action.no_op())
        
        # 2. Save the model
        model_path = os.path.join(tmp_path, "model.joblib")
        model.save(model_path)
        assert os.path.exists(model_path)
        
        # 3. Load into a NEW WorldModel instance
        new_model = WorldModel(rule=simple_stock_rule)
        new_model.load(model_path)
        
        # 4. Verify predictions are identical
        pred_after = new_model.predict(s_test, Action.no_op())
        
        np.testing.assert_array_almost_equal(pred_before.values, pred_after.values)
        assert new_model.correction_trained is True

    @pytest.mark.unit
    def test_save_untrained_model_raises_error(self, tmp_path):
        """Cannot save if not trained."""
        model = WorldModel(rule=simple_stock_rule)
        model_path = os.path.join(tmp_path, "untrained.joblib")
        with pytest.raises(ValueError, match="untrained"):
            model.save(model_path)
