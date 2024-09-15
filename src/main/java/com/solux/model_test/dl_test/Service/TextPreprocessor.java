package com.solux.model_test.dl_test.Service;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class TextPreprocessor {
    private static final Map<String, Integer> VOCAB = new HashMap<>();

    static {
        try (BufferedReader br = new BufferedReader(new FileReader("src/main/resources/vocab.txt"))) {
            String line;
            int index = 0;
            while ((line = br.readLine()) != null) {
                VOCAB.put(line, index++);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Map<String, OnnxTensor> preprocessText(String text, OrtEnvironment env, int maxLength) throws OrtException {
        List<Integer> inputIds = tokenizeAndMapToIds(text);
        while (inputIds.size() < maxLength) {
            inputIds.add(0); // Padding token ID
        }
        if (inputIds.size() > maxLength) {
            inputIds = inputIds.subList(0, maxLength); // Truncate
        }


        OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, new long[][]{inputIds.stream().mapToLong(i -> i).toArray()});

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputIdsTensor);

        return inputs;
    }

    private List<Integer> tokenizeAndMapToIds(String text) {
        List<Integer> tokens = new ArrayList<>();
        for (String word : text.split(" ")) {
            tokens.add(VOCAB.getOrDefault(word, VOCAB.get("[UNK]"))); // Tokenize
        }
        return tokens;
    }
}

