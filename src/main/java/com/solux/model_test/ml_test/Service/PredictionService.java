package com.solux.model_test.ml_test.Service;

import org.dmg.pmml.PMML;
import org.jpmml.evaluator.*;
import org.jpmml.model.PMMLUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import javax.xml.parsers.ParserConfigurationException;
import java.io.InputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class PredictionService {

    private static final Logger logger = LoggerFactory.getLogger(PredictionService.class);

    private Evaluator modelEvaluator;

    public PredictionService() throws IOException, JAXBException, SAXException, ParserConfigurationException {
        try (InputStream pmmlStream = getClass().getClassLoader().getResourceAsStream("ml_model.pmml")) {
            if (pmmlStream == null) {
                throw new IOException("Resource 'ml_model.pmml' not found in classpath");
            }

            PMML pmml = PMMLUtil.unmarshal(pmmlStream);
            this.modelEvaluator = new ModelEvaluatorBuilder(pmml).build();
            this.modelEvaluator.verify();
        } catch (jakarta.xml.bind.JAXBException e) {
            throw new RuntimeException(e);
        }
    }

    public Map<String, Object> predict(Map<String, Object> input) {
        Map<String, Object> response = new HashMap<>();

        try {
            Map<String, Object> transformedInput = new HashMap<>();
            transformedInput.put("Year", input.get("연도"));
            transformedInput.put("Month", input.get("월"));
            transformedInput.put("품목명", input.get("품목명"));
            transformedInput.put("품종명", input.get("품종명"));
            transformedInput.put("등급명", input.get("등급명"));
            transformedInput.put("유통단계별무게", input.get("유통단계별무게"));

            if (input.get("품목명") != null && input.get("품종명") != null) {
                transformedInput.put("품목_품종_상호작용", input.get("품목명") + "_" + input.get("품종명"));
            } else {
                transformedInput.put("품목_품종_상호작용", null);
            }

            List<InputField> inputFields = modelEvaluator.getInputFields();

            Map<String, FieldValue> arguments = new LinkedHashMap<>();
            for (InputField inputField : inputFields) {
                String fieldName = inputField.getName().toString();
                Object rawValue = transformedInput.get(fieldName);

                FieldValue fieldValue = inputField.prepare(rawValue);
                arguments.put(fieldName, fieldValue);
            }

            Map<String, ?> results = modelEvaluator.evaluate(arguments);

            for (Map.Entry<String, ?> entry : results.entrySet()) {
                if ("y".equals(entry.getKey())) {
                    Object value = entry.getValue();

                    if (value instanceof Map) {
                        Map<?, ?> nestedMap = (Map<?, ?>) value;
                        if (nestedMap.containsKey("result")) {
                            response.put("result", nestedMap.get("result"));
                        }
                    } else if (value instanceof Computable) {
                        Computable computable = (Computable) value;
                        response.put("result", computable.getResult());
                    }
                    break;
                }
            }

        } catch (Exception e) {
            response.put("error", e.getMessage());
        }

        return response;
    }
}

