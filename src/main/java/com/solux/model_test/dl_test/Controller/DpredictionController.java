package com.solux.model_test.dl_test.Controller;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.solux.model_test.dl_test.Service.TextPreprocessor;
import com.solux.model_test.dl_test.Service.UrlContentExtractor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@RestController
public class DpredictionController {

    private final TextPreprocessor textPreprocessor;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final UrlContentExtractor urlContentExtractor;

    public DpredictionController() throws OrtException, IOException {
        this.textPreprocessor = new TextPreprocessor();
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession("src/main/resources/dl_model.onnx", new OrtSession.SessionOptions());
        this.urlContentExtractor = new UrlContentExtractor();

        System.out.println("Model requires "+session.getInputInfo().size() +" inputs.");
        session.getInputInfo().forEach((name, info) -> {
            System.out.println("Input name: " + name);
            System.out.println("Input type: " + info.getInfo());
        });
    }
    @PostMapping("/dlpredict")
    public ResponseEntity<Map<String, Object>> predict(@RequestBody Map<String, String> request) {
        String url = request.get("url");
        String content;
        try {
            content = urlContentExtractor.extractContent(url);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(createErrorResponse("Failed to retrieve data from the URL"));
        }

        if (content == null || content.isEmpty()) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(createErrorResponse("No content retrieved from the URL"));
        }

        try {
            Map<String, OnnxTensor> inputs = textPreprocessor.preprocessText(content, env, 512);
            OrtSession.Result results = session.run(inputs);
            float[][] logits = (float[][]) results.get(0).getValue();

            System.out.println("Logits: ");
            for (float[] logit : logits) {
                System.out.println(Arrays.toString(logit));
            }
            int prediction = argmax(logits);

            return ResponseEntity.ok(Map.of("result", prediction));
        } catch (OrtException e) {
            e.printStackTrace(); // Consider using a logger
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(createErrorResponse("Model inference failed"));
        }
    }

    private int argmax(float[][] logits) {
        int maxIndex = 0;
        for (int i = 1; i < logits[0].length; i++) {
            if (logits[0][i] > logits[0][maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private Map<String, Object> createErrorResponse(String message) {
        Map<String, Object> errorResponse = new HashMap<>();
        errorResponse.put("error", message);
        return errorResponse;
    }
}


