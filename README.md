# SC Inspector

The **Spare Cores Inspector** is a key component of the [Spare
Cores](https://sparecores.com) project, designed to continuously monitor and
evaluate cloud instance types discovered through API integrations. This service
plays a vital role in ensuring that the latest cloud offerings are fully
analyzed and integrated into the published Spare Cores Navigator database.

## How It Works

1. **Monitoring**: Continuously listens for updates (such as new instance types or updates to existing instance types that might trigger new or updated benchmarks) via the [`sparecores-data`](https://pypi.org/project/sparecores-data/) package.
2. **Inspection**: When a new instance type is detected, the service initiates an inspection process to collect hardware data.
3. **Benchmarking**: Various benchmarks are executed to assess performance metrics. For a detailed list of benchmarks, see the [tasks.py](./inspector/tasks.py) file.
4. **Data Publishing**: All collected information is stored in a structured format in the [`sc-inspector-data`](https://github.com/SpareCores/sc-inspector-data) repository.
