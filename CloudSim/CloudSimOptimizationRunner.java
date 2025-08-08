package org.cloudbus.cloudsim.examples;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;
import org.cloudbus.cloudsim.Datacenter;
import org.cloudbus.cloudsim.DatacenterBroker;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;

import com.opencsv.CSVWriter;


public class CloudSimOptimizationRunner {

    public static void main(String[] args) {
        try {
            // Expecting 3 arguments: <mips> <pes> <ram>
            // if (args.length != 3) {
            //     System.err.println("Usage: java CloudSimOptimizationRunner <mips> <pes> <ram>");
            //     return;
            // }

            int mips = Integer.parseInt(System.getenv("VM_MIPS"));
            int pes = Integer.parseInt(System.getenv("VM_PES"));
            int ram = Integer.parseInt(System.getenv("VM_RAM"));
            System.out.println("[JAVA] MIPS=" + mips + " PEs=" + pes + " RAM=" + ram);

            CloudSim.init(1, Calendar.getInstance(), false);

            Datacenter dc = CloudSimSimulationSuite.createDatacenterPublic("Datacenter_Opt");
            DatacenterBroker broker = new DatacenterBroker("Broker_Opt");
            int brokerId = broker.getId();


            // Create and submit VM
            Vm vm = new Vm(0, brokerId, mips, pes, ram, 10000, 20000,
               "Xen", new CloudletSchedulerSpaceShared());
            broker.submitGuestList(Collections.singletonList(vm));

            // Load and submit jobs
            List<TimeRangeJobLoader.JobInfo> jobs = TimeRangeJobLoader.fetchJobsByTimeRange(
                    "Monday", "9:00", "9:10", 100);

            CloudSimSimulationSuite.initializeUserTrackingPublic(jobs);
            CloudSimSimulationSuite.submitJobs(jobs, brokerId, dc.getId(), vm);

            // Start and stop simulation
            CloudSim.startSimulation();
            CloudSim.stopSimulation();

            List<Cloudlet> cloudlets = broker.getCloudletReceivedList();
            CloudSimSimulationSuite.computeAndPrintOverallMetricsPublic(cloudlets);

            CloudSimSimulationSuite.writeCsv(cloudlets);


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void writeSimulationMetricsToCSV(List<Cloudlet> cloudlets) {
        try {
            double makespan = cloudlets.stream()
                .mapToDouble(Cloudlet::getFinishTime)
                .max().orElse(0.0);

            double cost = cloudlets.stream()
                .mapToDouble(Cloudlet::getProcessingCost)
                .sum();

            double utilization = cloudlets.stream()
                .mapToDouble(cl -> cl.getActualCPUTime() / (cl.getFinishTime() - cl.getExecStartTime() + 0.0001))
                .average().orElse(0.0);

            CSVWriter writer = new CSVWriter(
                new FileWriter("C:/Users/blekh/cloudsim-master/modules/cloudsim-examples/simulation_metrics.csv")
            );

            writer.writeNext(new String[]{"Makespan", "Cost", "Utilization"});
            writer.writeNext(new String[]{
                String.valueOf(makespan),
                String.valueOf(cost),
                String.valueOf(utilization)
            });

            writer.close();
            System.out.println("[JAVA] simulation_metrics.csv written successfully.");
            System.out.println("[JAVA] Makespan=" + makespan + ", Cost=" + cost + ", Utilization=" + utilization);
        } catch (IOException e) {
            System.err.println("[JAVA] Error writing simulation_metrics.csv");
            e.printStackTrace();
        }
    }


}
