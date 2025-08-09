# ☁ Cloud Workload Simulation and Optimization Framework

This project provides a complete framework for simulating and optimizing cloud computing environments. It uses a sophisticated machine learning model, a *Variational Autoencoder (VAE), to generate realistic, synthetic job workloads. These workloads are then used in a **CloudSim*-based simulation to test and find the most efficient virtual machine (VM) configurations.

This is a great tool if you're a researcher, student, or developer working on cloud resource management, scheduling algorithms, or performance optimization. It lets you test your ideas in a controlled, simulated environment without needing expensive real-world hardware.



## What it Does

The project is broken down into a few key stages that work together:

1.  *Analyzes Real Workloads* 📊: The process starts by taking a real-world workload log file (in .swf format) and analyzing it. A Python script (Synthesizer/swf_utils/swf_categorizer3.py) cleans the data and categorizes jobs based on their resource needs (like CPU and runtime) and how "bursty" they are (how frequently they're submitted). It also profiles users based on their submission patterns.

2.  *Generates Synthetic Jobs with AI* 🤖: This categorized job data is then used to train a *Variational Autoencoder (VAE)*, which is a type of AI model. The VAE learns the patterns in the real job data and can then generate new, synthetic jobs that are statistically similar. This is handled by the Synthesizer/train_all_vae2.py script.

3.  *Serves Jobs via an API* ☁: A simple web server, built with *Flask*, is used to serve these synthetic jobs. The CloudSim simulation environment can then request a stream of jobs for a specific time of day, like "Monday from 9 AM to 11 AM." This is all managed by Synthesizer/simulation_with_inter.py.

4.  *Simulates the Cloud Environment* ⚙: The main simulation is built with *CloudSim*, a Java-based framework. The CloudSim/CloudSimSimulationSuite.java file sets up the simulated datacenter, hosts, and VMs. It then gets the synthetic jobs from the Flask API and runs them on the simulated hardware.

5.  *Optimizes and Visualizes Results* 📈: The project uses *Bayesian Optimization* to automatically find the best VM configuration. It runs the simulation many times with different settings for the VM's processing power (MIPS), number of cores (PEs), and RAM. The goal is to find the settings that minimize job waiting time. The results can then be visualized to see how different configurations performed.


### Key Technologies

* *CloudSim*: A Java framework for building and testing cloud computing systems. It allows you to model datacenters, hosts, VMs, and job scheduling without needing physical hardware.
* *Variational Autoencoder (VAE)*: A type of AI model that's great for learning the underlying structure of data and generating new, similar data. In this project, it's used to create realistic job workloads for the simulation.
* *Flask*: A lightweight Python framework for building web apps and APIs. Here, it's used to create the server that provides synthetic jobs to the CloudSim simulation.
* *Bayesian Optimization*: An efficient strategy for finding the best solution to a problem by building a probabilistic model of it. It's used to find the most efficient VM hardware configuration.


## Setup and Installation
### Prerequisites

* *Java*: Make sure you have a recent version of the Java Development Kit (JDK) installed.
* *Maven*: This is used to build the Java parts of the project.
* *Python*: You'll need Python 3. You can check the requirements.txt file for the specific libraries used.


### Step-by-Step Guide

1.  *Clone the Repository*

    ```bash

    git clone https://github.com/PoornavG/Vae-flask.git

    cd Vae-flask

    ```



2.  *Set up the Python Environment*

    It's a good idea to use a virtual environment to keep your project's dependencies separate.



    ```bash

    python -m venv venv

    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    

    Then, install the required Python libraries:

    ```bash

    pip install -r requirements.txt
    ```
    

3.  *Configure the Project*

    The Synthesizer/config.py file holds all the important settings for the project. The most important one to check is SWF_PATH, which should point to the location of your workload log file.

    ```python

    # In Synthesizer/config.py

    SWF_PATH = r"/path/to/your/workload.swf"

    ```


4. Clone CloudSim 5.0

If you haven't already:
```bash
git clone https://github.com/Cloudslab/cloudsim.git cloudsim-master
```

5. Place Java Files in Correct Package Path

Copy the following files:

- `CloudSimOptimizationRunner.java`
- `CloudSimSimulationSuite.java`
- `TimeRangeJobLoader.java`

**To this path:**
cloudsim-master/modules/cloudsim-examples/src/main/java/org/cloudbus/cloudsim/examples/

This is required because these classes belong to the `org.cloudbus.cloudsim.examples` Java package.


6. Place Python Scripts in Module Root

Copy:

- `optimize_cloudsim.py`
- `visualize_optimization.py`

**To:**
cloudsim-master/modules/cloudsim-examples/
These scripts run from the CloudSim examples root and interact with the Java simulation via CLI and CSV files.


5.  *Initial Data Processing and VAE Training*

    Before you can run the simulation, you need to process your workload data and train the VAE model. The simulation_with_inter.py script has an interactive setup that will guide you through this.

    ```bash

    python Synthesizer/simulation_with_inter.py

    ```

    The first time you run this, it will ask if you want to:

    * *Run data categorization: Say **yes* to this. It will process your .swf file, create user profiles, and analyze the job distribution.

    * *Train the VAE models: Say **yes* to this as well. This will train the VAE on the categorized data. This can take some time.



    After this is done, you'll have a vae_models directory with the trained AI models and some .json and .csv files with the data analysis results.



## Running the Project

Once everything is set up, you can run the different parts of the project.

1.  *Start the Flask Job Generation Server*

    First, you need to start the Flask server so that CloudSim can get the synthetic jobs.

    ```bash

    python Synthesizer/simulation_with_inter.py
    ```
    
    This will start the server, and you should see a message that it's running. Keep this terminal window open.


2.  *Run a Single CloudSim Simulation*

Navigate to the CloudSim examples module and compile:

```bash
cd cloudsim-master/modules/cloudsim-examples
mvn clean compile
```

To run a simulation manually:

```bash
mvn exec:java -Dexec.mainClass="org.cloudbus.cloudsim.examples.CloudSimOptimizationRunner"
```



3.  *Find the Best VM Configuration*

    To automatically find the best VM configuration, you can use the Bayesian Optimization script. This will run the simulation many times with different hardware settings.

    ```bash

    # Make sure you're in the CloudSim directory

    python optimize_cloudsim.py
    ```
    
    This will take some time, as it's running many simulations. It will print out the results of each run and will tell you the best configuration it found at the end.



4.  *Visualize the Optimization Results*

    After the optimization is complete, a file named optimization_log.csv will be created. You can use the visualize_optimization.py script to see how the optimization performed.

    ```bash

    # Make sure you're in the CloudSim directory

    python visualize_optimization.py
    ```
    

    This will generate plots showing how the objective function changed over time and a heatmap showing the correlation between the different hardware parameters and performance.
