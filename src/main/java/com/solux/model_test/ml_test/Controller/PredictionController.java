package com.solux.model_test.ml_test.Controller;

import com.solux.model_test.ml_test.Service.PredictionService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
public class PredictionController {

    private final PredictionService predictionService;

    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @PostMapping("/predict")
    public Map<String, Object> predict(@RequestBody Map<String, Object> input) {
        return predictionService.predict(input);
    }
}
