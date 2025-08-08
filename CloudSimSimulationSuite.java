package org.cloudbus.cloudsim.examples;

import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;
import org.cloudbus.cloudsim.Datacenter;
import org.cloudbus.cloudsim.DatacenterBroker;
import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.VmAllocationPolicySimple;
import org.cloudbus.cloudsim.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.core.CloudActionTags;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;

public class CloudSimSimulationSuite {

    // Keep track of original JobInfo and arrival times for metrics
    private static final Map<Integer, TimeRangeJobLoader.JobInfo> jobInfoMap = new HashMap<>();
    private static java.util.Map<String, Double> lastSimulationMetrics = new java.util.HashMap<>();

    private static final Map<Integer, Double> arrivalMap = new HashMap<>();

    // Track user burst management - now properly per-user
    private static final Map<Integer, String> userBurstLevels = new HashMap<>();
    private static final Map<Integer, Integer> userLongWaitingJobs = new HashMap<>();
    private static final Map<Integer, Boolean> userDropped = new HashMap<>();
    private static final Map<Integer, Double> userInterarrivalAdjustment = new HashMap<>();

    // Correct thresholds for waiting time (in seconds)
    private static final int HIGH_BURST_THRESHOLD = 5000; // 10000s for high burst users
    private static final int MED_BURST_THRESHOLD = 10000; // 20000s for medium burst users
    private static final int MAX_WAITING_JOBS = 3;
    private static final double INTERARRIVAL_PENALTY = 600.0; // 600s penalty

    // Track submitted and dropped jobs for statistics
    private static int totalJobsDropped = 0;
    private static final Map<Integer, Integer> droppedJobsPerUser = new HashMap<>();

    public static void main(String[] args) {
        try {
            Log.println("Starting CloudSimSimulationSuite...");
            CloudSim.init(1, Calendar.getInstance(), false);

            // 1) Create Datacenter with 150 cores
            Datacenter dc = createDatacenter("Datacenter_0");
            writeDatacenterDetails(dc);
            writeHostDetails(dc.getHostList());

            // 2) Create Broker and single VM with 120 MIPS, 150 PEs, SpaceShared scheduler
            DatacenterBroker broker = new DatacenterBroker("Broker");
            int brokerId = broker.getId();
            Vm vm = createAndSubmitSingleVm(broker, brokerId);
            writeVmDetails(vm);

            // 3) Fetch jobs using TimeRangeJobLoader (hits /simulate_by_time_range)
            List<TimeRangeJobLoader.JobInfo> jobs = TimeRangeJobLoader.fetchJobsByTimeRange(
                    "Monday", // day
                    "9:00", // start
                    "9:10", // end
                    100 // total jobs to request
            );
            printJobDetails(jobs);

            // 4) Initialize user tracking
            initializeUserTracking(jobs);

            // 5) Dynamic acceptance + scheduling with burst management
            final double BASE_DELAY = 0.05;
            double cumulativeDelay = BASE_DELAY;
            int id = 0;
            int submittedJobs = 0;

            for (TimeRangeJobLoader.JobInfo j : jobs) {
                if (j.length <= 0 && j.cpuTime <= 0) {
                    Log.println("Skipping invalid job " + id + " (length=0 && cpuTime=0)");
                    id++;
                    continue;
                }
                if (j.length <= 0)
                    j.length = j.cpuTime;
                if (j.cpuTime <= 0)
                    j.cpuTime = j.length;

                // Cap requested PEs to VM max (150)
                j.pes = Math.min(j.pes, 150);

                // Check if user should be dropped
                if (userDropped.getOrDefault(j.userId, false)) {
                    Log.println(String.format("Job %d from user %d DROPPED - user has been blocked",
                            id, j.userId));
                    droppedJobsPerUser.put(j.userId, droppedJobsPerUser.getOrDefault(j.userId, 0) + 1);
                    totalJobsDropped++;
                    id++;
                    continue;
                }

                // Apply interarrival time adjustment if user has penalty
                double adjustedInterarrival = j.interarrival + userInterarrivalAdjustment.getOrDefault(j.userId, 0.0);

                // Store original for CSV
                TimeRangeJobLoader.JobInfo info = new TimeRangeJobLoader.JobInfo();
                info.length = j.length;
                info.cpuTime = j.cpuTime;
                info.interarrival = j.interarrival; // Store original
                info.pes = j.pes;
                info.category = j.category;
                info.userId = j.userId;
                info.userBurstLevel = j.userBurstLevel;
                jobInfoMap.put(id, info);

                // Calculate MI = runtime_seconds × VM_MIPS × PEs
                long mi = Math.round(j.cpuTime * vm.getMips() * j.pes);

                // Create cloudlet with calculated MI
                Cloudlet c = new Cloudlet(
                        id,
                        mi, // use computed MI
                        Math.max(1, j.pes),
                        300, 300,
                        new UtilizationModelFull(),
                        new UtilizationModelFull(),
                        new UtilizationModelFull());
                c.setUserId(brokerId);
                c.setGuestId(vm.getId());

                arrivalMap.put(id, cumulativeDelay);

                CloudSim.send(
                        brokerId,
                        dc.getId(),
                        cumulativeDelay,
                        CloudActionTags.CLOUDLET_SUBMIT,
                        c);

                Log.println(String.format("Job %d submitted for user %d (burst: %s, adjusted interarrival: %.2f)",
                        id, j.userId, j.userBurstLevel, adjustedInterarrival));

                cumulativeDelay += adjustedInterarrival; // Use adjusted interarrival
                submittedJobs++;
                id++;
            }

            Log.println(String.format("Total jobs processed: %d, Submitted: %d, Dropped: %d",
                    jobs.size(), submittedJobs, totalJobsDropped));

            CloudSim.startSimulation();
            CloudSim.stopSimulation();

            List<Cloudlet> finished = broker.getCloudletReceivedList();
            processSimulationResults(finished);
            printCloudletList(finished);
            writeCsv(finished);
            writeBrokerDetails(broker, submittedJobs);
            computeAndPrintOverallMetrics(finished);
            printUserStatistics();

            Log.println("CloudSimSimulationSuite finished!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Datacenter createDatacenterPublic(String name) throws Exception {
        return createDatacenter(name); // The original method is private
    }

    public static void initializeUserTracking(List<TimeRangeJobLoader.JobInfo> jobs) {
        for (TimeRangeJobLoader.JobInfo j : jobs) {
            if (!userBurstLevels.containsKey(j.userId)) {
                userBurstLevels.put(j.userId, j.userBurstLevel);
                userLongWaitingJobs.put(j.userId, 0);
                userDropped.put(j.userId, false);
                userInterarrivalAdjustment.put(j.userId, 0.0);
                droppedJobsPerUser.put(j.userId, 0);
            }
        }
        Log.println(String.format("Initialized tracking for %d unique users", userBurstLevels.size()));
    }

    public static int submitJobs(List<TimeRangeJobLoader.JobInfo> jobs, int brokerId, int datacenterId, Vm vm) {
        final double BASE_DELAY = 0.05;
        double cumulativeDelay = BASE_DELAY;
        int submittedJobs = 0;
        int id = 0;

        for (TimeRangeJobLoader.JobInfo j : jobs) {
            if (j.length <= 0 && j.cpuTime <= 0) {
                Log.println("Skipping invalid job " + id + " (length=0 && cpuTime=0)");
                id++;
                continue;
            }
            if (j.length <= 0)
                j.length = j.cpuTime;
            if (j.cpuTime <= 0)
                j.cpuTime = j.length;

            j.pes = Math.min(j.pes, 150);

            if (userDropped.getOrDefault(j.userId, false)) {
                Log.println(String.format("Job %d from user %d DROPPED - user has been blocked", id, j.userId));
                droppedJobsPerUser.put(j.userId, droppedJobsPerUser.getOrDefault(j.userId, 0) + 1);
                totalJobsDropped++;
                id++;
                continue;
            }

            double adjustedInterarrival = j.interarrival + userInterarrivalAdjustment.getOrDefault(j.userId, 0.0);

            // Track job info
            TimeRangeJobLoader.JobInfo info = new TimeRangeJobLoader.JobInfo();
            info.length = j.length;
            info.cpuTime = j.cpuTime;
            info.interarrival = j.interarrival;
            info.pes = j.pes;
            info.category = j.category;
            info.userId = j.userId;
            info.userBurstLevel = j.userBurstLevel;
            jobInfoMap.put(id, info);

            long mi = Math.round(j.cpuTime * vm.getMips() * j.pes);

            Cloudlet c = new Cloudlet(
                    id,
                    mi,
                    Math.max(1, j.pes),
                    300, 300,
                    new UtilizationModelFull(),
                    new UtilizationModelFull(),
                    new UtilizationModelFull());

            c.setUserId(brokerId);
            c.setGuestId(vm.getId());
            arrivalMap.put(id, cumulativeDelay);

            CloudSim.send(brokerId, datacenterId, cumulativeDelay, CloudActionTags.CLOUDLET_SUBMIT, c);

            Log.println(String.format("Job %d submitted for user %d (burst: %s, adjusted interarrival: %.2f)",
                    id, j.userId, j.userBurstLevel, adjustedInterarrival));

            cumulativeDelay += adjustedInterarrival;
            submittedJobs++;
            id++;
        }

        return submittedJobs;
    }

    public static void initializeUserTrackingPublic(List<TimeRangeJobLoader.JobInfo> jobs) {
        initializeUserTracking(jobs);
    }

    public static void computeAndPrintOverallMetricsPublic(List<Cloudlet> cloudlets) {
        double makespan = 0;
        double totalCost = 0;
        double utilization = 0;

        for (Cloudlet c : cloudlets) {
            makespan = Math.max(makespan, c.getFinishTime());
            totalCost += c.getActualCPUTime() * c.getNumberOfPes(); // Simplified cost model
            utilization += c.getActualCPUTime();
        }

        utilization = utilization / makespan;

        System.out.println("===== SIMULATION METRICS =====");
        System.out.println("Makespan: " + makespan);
        System.out.println("Cost: " + totalCost);
        System.out.println("Utilization: " + utilization);

        lastSimulationMetrics.put("makespan", makespan);
        lastSimulationMetrics.put("cost", totalCost);
        lastSimulationMetrics.put("utilization", utilization);
    }

    public static java.util.Map<String, Double> getLastSimulationMetrics() {
        return lastSimulationMetrics;
    }


    public static Map<String, Double> computeMetrics(List<Cloudlet> cloudlets) {
        int count = 0;
        double sumWait = 0, sumRun = 0;
        double firstArrival = Double.MAX_VALUE, lastFinish = 0;

        for (Cloudlet c : cloudlets) {
            if (c.getStatus() != Cloudlet.CloudletStatus.SUCCESS)
                continue;

            int id = c.getCloudletId();
            TimeRangeJobLoader.JobInfo info = jobInfoMap.get(id);
            double arrival = arrivalMap.get(id);
            double start = c.getExecStartTime();
            double finish = c.getExecFinishTime();
            double runtime = finish - start;
            double waiting = start - arrival;

            count++;
            sumRun += runtime;
            sumWait += waiting;
            firstArrival = Math.min(firstArrival, arrival);
            lastFinish = Math.max(lastFinish, finish);
        }

        double makespan = lastFinish - firstArrival;
        double throughput = count / makespan;
        double avgRun = sumRun / count;
        double avgWait = sumWait / count;

        Map<String, Double> metrics = new HashMap<>();
        metrics.put("makespan", makespan);
        metrics.put("avg_runtime", avgRun);
        metrics.put("avg_waiting", avgWait);
        metrics.put("throughput", throughput);
        metrics.put("count", (double) count);

        // Optional: write to CSV
        try (PrintWriter pw = new PrintWriter("optimization_metrics.csv")) {
            pw.println("makespan,avg_runtime,avg_waiting,throughput,count");
            pw.printf(Locale.US, "%.4f,%.4f,%.4f,%.4f,%.0f%n",
                    makespan, avgRun, avgWait, throughput, (double) count);
        } catch (Exception e) {
            Log.printLine("Error writing optimization_metrics.csv: " + e.getMessage());
        }

        return metrics;
    }



    private static Vm createAndSubmitSingleVm(DatacenterBroker broker, int brokerId) {
        Vm vm = new Vm(
                0, brokerId,
                120, 150, // MIPS, PEs = 150
                8192, // RAM (MB)
                10000, // BW
                20000, // size (MB)
                "Xen",
                new CloudletSchedulerSpaceShared()); // SpaceShared scheduler
        broker.submitGuestList(Collections.singletonList(vm));
        return vm;
    }

    private static void printJobDetails(List<TimeRangeJobLoader.JobInfo> jobs) {
        System.out.println("\n=== Jobs fetched ===");
        int i = 0;
        for (TimeRangeJobLoader.JobInfo j : jobs) {
            System.out.printf(
                    "Job %d: length(raw)=%.2f cpuTime(s)=%.2f inter=%.2f pes=%d  user=%d burst=%s%n",
                    i++, j.length, j.cpuTime, j.interarrival, j.pes, j.userId, j.userBurstLevel);
        }
    }

    public static Datacenter createDatacenter(String name) throws Exception {
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < 150; i++) {
            peList.add(new Pe(i, new PeProvisionerSimple(1000)));
        }

        Host host = new Host(
                0,
                new RamProvisionerSimple(32768),
                new BwProvisionerSimple(10000),
                1_000_000,
                peList,
                new VmSchedulerTimeShared(peList));

        List<Host> hostList = new ArrayList<>();
        hostList.add(host);

        DatacenterCharacteristics chr = new DatacenterCharacteristics(
                "x86", "Linux", "Xen",
                hostList,
                10.0, 3.0, 0.05, 0.001, 0.0);

        return new Datacenter(
                name, chr,
                new VmAllocationPolicySimple(hostList),
                new LinkedList<>(),
                0);
    }

    public static Vm createCustomVm(int vmId, int mips, int pes, int ram) {
        return new Vm(
                vmId, 0, // brokerId will be set when submitted to broker
                mips, pes,
                ram,
                10000, // BW
                10000, // Size (MB)
                "Xen",
                new CloudletSchedulerSpaceShared()
        );
    }


    private static void printCloudletList(List<Cloudlet> list) {
        DecimalFormat dft = new DecimalFormat("###.##");
        Log.println("\n=== OUTPUT ===");
        Log.println("ID  Status    DC VM PEs Time    Start   Finish  Waiting");
        for (Cloudlet c : list) {
            if (c.getStatus() == Cloudlet.CloudletStatus.SUCCESS) {
                double arrival = arrivalMap.get(c.getCloudletId());
                double waiting = c.getExecStartTime() - arrival;
                Log.println(String.format(
                        "%3d   %-7s   %3d %2d %3d %6.2f   %6.2f   %6.2f   %6.2f",
                        c.getCloudletId(), "SUCCESS",
                        c.getResourceId(), c.getVmId(), c.getNumberOfPes(),
                        c.getActualCPUTime(),
                        c.getExecStartTime(),
                        c.getExecFinishTime(),
                        waiting));
            }
        }
    }

    public static void writeCsv(List<Cloudlet> list) {
        try (PrintWriter pw = new PrintWriter("simulation_metrics.csv")) {
            pw.println(
                    "id,length,cpuTime,pes,arrival,start,finish,runtime,waiting,category,userId,burstLevel,userDropped,userPenalty");
            for (Cloudlet c : list) {
                if (c.getStatus() != Cloudlet.CloudletStatus.SUCCESS)
                    continue;
                int id = c.getCloudletId();
                TimeRangeJobLoader.JobInfo info = jobInfoMap.get(id);
                double arrival = arrivalMap.get(id);
                double start = c.getExecStartTime();
                double finish = c.getExecFinishTime();
                double runtime = finish - start;
                double waiting = start - arrival;
                boolean isUserDropped = userDropped.getOrDefault(info.userId, false);
                double userPenalty = userInterarrivalAdjustment.getOrDefault(info.userId, 0.0);

                pw.printf(Locale.US,
                        "%d,%d,%.2f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%s,%d,%s,%b,%.2f%n",
                        id,
                        c.getCloudletLength(),
                        info.cpuTime,
                        info.pes,
                        arrival,
                        start,
                        finish,
                        runtime,
                        waiting,
                        info.category,
                        info.userId,
                        info.userBurstLevel,
                        isUserDropped,
                        userPenalty);
            }
            Log.println("CSV written to simulation_metrics.csv");
        } catch (Exception e) {
            Log.println("Error writing CSV: " + e.getMessage());
        }
    }

    private static void processSimulationResults(List<Cloudlet> finished) {
        Log.println("\n=== Processing Simulation Results for Burst Management ===");

        for (Cloudlet c : finished) {
            if (c.getStatus() != Cloudlet.CloudletStatus.SUCCESS)
                continue;

            int jobId = c.getCloudletId();
            TimeRangeJobLoader.JobInfo info = jobInfoMap.get(jobId);
            double arrival = arrivalMap.get(jobId);
            double start = c.getExecStartTime();
            double waiting = start - arrival;

            int userId = info.userId;
            String burstLevel = info.userBurstLevel;

            // Check if this job had excessive waiting time
            boolean longWait = false;
            if ("High".equals(burstLevel) && waiting > HIGH_BURST_THRESHOLD) {
                longWait = true;
                Log.println(String.format("High burst user %d job %d exceeded threshold: %.2f > %d",
                        userId, jobId, waiting, HIGH_BURST_THRESHOLD));
            } else if ("Medium".equals(burstLevel) && waiting > MED_BURST_THRESHOLD) {
                longWait = true;
                Log.println(String.format("Medium burst user %d job %d exceeded threshold: %.2f > %d",
                        userId, jobId, waiting, MED_BURST_THRESHOLD));
            }

            if (longWait) {
                // Increment long waiting jobs count for this user
                int currentCount = userLongWaitingJobs.get(userId);
                userLongWaitingJobs.put(userId, currentCount + 1);

                // Apply interarrival penalty
                double currentPenalty = userInterarrivalAdjustment.getOrDefault(userId, 0.0);
                userInterarrivalAdjustment.put(userId, currentPenalty + INTERARRIVAL_PENALTY);

                Log.println(String.format("User %d now has %d long-waiting jobs, penalty increased to %.2f",
                        userId, currentCount + 1, currentPenalty + INTERARRIVAL_PENALTY));

                // Check if user should be dropped
                if (currentCount + 1 > MAX_WAITING_JOBS) {
                    userDropped.put(userId, true);
                    Log.println(String.format("User %d BLOCKED - exceeded %d long-waiting jobs threshold",
                            userId, MAX_WAITING_JOBS));
                }
            }
        }
    }

    private static void printUserStatistics() {
        Log.println("\n=== User Burst Management Statistics ===");
        Log.println("UserID | BurstLevel | LongWaitJobs | Penalty | Dropped | JobsDropped");
        Log.println("-------|------------|--------------|---------|---------|------------");

        for (Integer userId : userBurstLevels.keySet()) {
            String burstLevel = userBurstLevels.get(userId);
            int longWaitJobs = userLongWaitingJobs.get(userId);
            double penalty = userInterarrivalAdjustment.get(userId);
            boolean dropped = userDropped.get(userId);
            int jobsDropped = droppedJobsPerUser.get(userId);

            Log.println(String.format("%6d | %-10s | %12d | %7.0f | %7s | %11d",
                    userId, burstLevel, longWaitJobs, penalty, dropped, jobsDropped));
        }

        Log.println(String.format("\nTotal jobs dropped: %d", totalJobsDropped));
    }

    private static void computeAndPrintOverallMetrics(List<Cloudlet> list) {
        int count = 0;
        double sumWait = 0, sumRun = 0;
        double firstArrival = Double.MAX_VALUE, lastFinish = 0;
        int longWaitCount = 0;

        for (Cloudlet c : list) {
            if (c.getStatus() != Cloudlet.CloudletStatus.SUCCESS)
                continue;
            int id = c.getCloudletId();
            TimeRangeJobLoader.JobInfo info = jobInfoMap.get(id);
            double arrival = arrivalMap.get(id);
            double start = c.getExecStartTime();
            double finish = c.getExecFinishTime();
            double runtime = finish - start;
            double waiting = start - arrival;

            count++;
            sumRun += runtime;
            sumWait += waiting;
            firstArrival = Math.min(firstArrival, arrival);
            lastFinish = Math.max(lastFinish, finish);

            // Count long waiting jobs
            if (("High".equals(info.userBurstLevel) && waiting > HIGH_BURST_THRESHOLD) ||
                    ("Medium".equals(info.userBurstLevel) && waiting > MED_BURST_THRESHOLD)) {
                longWaitCount++;
            }
        }

        double makespan = lastFinish - firstArrival;
        double throughput = count / makespan;
        double avgRun = sumRun / count;
        double avgWait = sumWait / count;

        Log.println("\n=== Overall Metrics ===");
        Log.println("Valid cloudlets : " + count);
        Log.println("Long waiting jobs: " + longWaitCount);
        Log.println("Jobs dropped    : " + totalJobsDropped);
        Log.println(String.format(Locale.US,
                "Avg waiting     : %.2f s\n" +
                        "Avg runtime     : %.2f s\n" +
                        "Makespan        : %.2f s\n" +
                        "Throughput      : %.4f jobs/s",
                avgWait, avgRun, makespan, throughput));
    }

    private static void writeVmDetails(Vm vm) {
        try (PrintWriter pw = new PrintWriter("vm_details.csv")) {
            pw.println("vmId,brokerId,mips,pes,ram,bw,size,vmm,scheduler");
            pw.printf(Locale.US, "%d,%d,%d,%d,%d,%d,%d,%s,%s%n",
                    vm.getId(), vm.getUserId(), vm.getMips(), vm.getNumberOfPes(),
                    vm.getRam(), vm.getBw(), vm.getSize(),
                    vm.getVmm(), vm.getCloudletScheduler().getClass().getSimpleName());
        } catch (Exception e) {
            Log.println("Error writing VM details: " + e.getMessage());
        }
    }

    private static void writeBrokerDetails(DatacenterBroker broker, int submittedCloudlets) {
        try (PrintWriter pw = new PrintWriter("broker_details.csv")) {
            pw.println("brokerId,name,submittedCloudlets,totalDropped");
            pw.printf(Locale.US, "%d,%s,%d,%d%n",
                    broker.getId(), broker.getName(), submittedCloudlets, totalJobsDropped);
        } catch (Exception e) {
            Log.println("Error writing Broker details: " + e.getMessage());
        }
    }

    private static void writeDatacenterDetails(Datacenter dc) {
        try (PrintWriter pw = new PrintWriter("datacenter_details.csv")) {
            java.lang.reflect.Method getChr = Datacenter.class
                    .getDeclaredMethod("getCharacteristics");
            getChr.setAccessible(true);
            DatacenterCharacteristics dcc = (DatacenterCharacteristics) getChr.invoke(dc);

            java.lang.reflect.Method getArch = DatacenterCharacteristics.class
                    .getDeclaredMethod("getArchitecture");
            getArch.setAccessible(true);
            String arch = (String) getArch.invoke(dcc);

            java.lang.reflect.Method getOs = DatacenterCharacteristics.class
                    .getDeclaredMethod("getOs");
            getOs.setAccessible(true);
            String os = (String) getOs.invoke(dcc);

            java.lang.reflect.Method getVmm = DatacenterCharacteristics.class
                    .getDeclaredMethod("getVmm");
            getVmm.setAccessible(true);
            String vmm = (String) getVmm.invoke(dcc);

            int numHosts = dcc.getHostList().size();
            double costSec = dcc.getCostPerSecond();
            double costMem = dcc.getCostPerMem();
            double costStore = dcc.getCostPerStorage();
            double costBw = dcc.getCostPerBw();

            pw.println("name,arch,os,vmm,numHosts,costPerSec,costPerMem,costPerStorage,costPerBw");
            pw.printf(Locale.US, "%s,%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f%n",
                    dc.getName(), arch, os, vmm,
                    numHosts, costSec, costMem, costStore, costBw);

        } catch (Exception e) {
            Log.println("Error writing Datacenter details: " + e.getMessage());
        }
    }

    private static void writeHostDetails(List<Host> hosts) {
        try (PrintWriter pw = new PrintWriter("host_details.csv")) {
            pw.println("hostId,ram,bw,storage,numPes,peMips,scheduler");
            for (Host h : hosts) {
                pw.printf(Locale.US, "%d,%d,%d,%d,%d,%d,%s%n",
                        h.getId(), h.getRamProvisioner().getRam(),
                        h.getBwProvisioner().getBw(), h.getStorage(),
                        h.getNumberOfPes(), h.getPeList().get(0).getMips(),
                        h.getVmScheduler().getClass().getSimpleName());
            }
        } catch (Exception e) {
            Log.println("Error writing Host details: " + e.getMessage());
        }
    }
}