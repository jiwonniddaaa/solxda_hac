package com.solux.model_test.dl_test.Service;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

public class UrlContentExtractor {

    public String extractContent(String url) {
        try {
            Document doc = Jsoup.connect(url).get();
            String title = doc.title();
            String description = doc.select("meta[name=description]").attr("content");
            return "title: " + title + ", desc: " + description;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
