package org.cloudbus.cloudsim.examples;

import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;

public class TimeRangeJobLoader {
    private static final String BASE_URL = "http://127.0.0.1:5000/simulate_by_time_range";
    private static final HttpClient CLIENT = HttpClient.newHttpClient();
    private static final Gson GSON = new Gson();

    public static class JobInfo {
        public double length;
        @SerializedName("cpu_time")
        public double cpuTime;
        public double interarrival;
        public int pes;
        public double ram;

        @SerializedName("user_id")
        public Integer userId;
        @SerializedName("user_burst_level")
        public String userBurstLevel;

        @SerializedName("category")
        public String category;

        public boolean isFromBurstyUser() {
            return "High".equals(userBurstLevel);
        }

        public boolean hasUserContext() {
            return userId != null;
        }
    }

    /**
     * Fetch jobs for a specific day and time range using Flask's time-based
     * simulation API.
     */
    public static List<JobInfo> fetchJobsByTimeRange(String day, String startTime, String endTime, int totalJobCount)
            throws Exception {

        String requestBody = String.format(
                "{\"day\": \"%s\", \"start_time\": \"%s\", \"end_time\": \"%s\", \"total_job_count\": %d}",
                day, startTime, endTime, totalJobCount);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(BASE_URL))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<java.io.InputStream> response = CLIENT.send(request, HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() != 200) {
            throw new RuntimeException("Failed to fetch jobs: " + response.statusCode());
        }

        InputStreamReader reader = new InputStreamReader(response.body());
        JobInfo[] jobs = GSON.fromJson(reader, JobInfo[].class);

        List<JobInfo> jobList = new ArrayList<>();
        for (JobInfo job : jobs) {
            jobList.add(job);
        }

        return jobList;
    }

    // For demo/testing
    public static void main(String[] args) {
        try {
            List<JobInfo> jobs = fetchJobsByTimeRange("Monday", "09:00", "11:00", 500);
            System.out.println("Fetched " + jobs.size() + " jobs:");
            for (int i = 0; i < Math.min(jobs.size(), 10); i++) {
                JobInfo job = jobs.get(i);
                System.out.printf("Job %d - Length: %.2f, CPU Time: %.2f, RAM: %.2f, PEs: %d, Category: %s, User: %s%n",
                        i + 1, job.length, job.cpuTime, job.ram, job.pes, job.category,
                        job.userId != null ? job.userId.toString() : "N/A");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}