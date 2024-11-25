# SC Inspector

**SC Inspector** is a key component of the [Spare Cores](https://sparecores.com) project, designed to continuously
monitor and process new cloud instance types discovered through API integrations.
This service plays a vital role in ensuring that the latest cloud offerings are fully analyzed and integrated 
into the published database.

## How It Works

1. **Monitoring**: SC Inspector continuously listens for updates via the [`sparecores-data`](https://pypi.org/project/sparecores-data/) package.
2. **Inspection**: When a new instance type is detected, the service initiates an inspection process to collect hardware data.
3. **Benchmarking**: Various benchmarks are executed to assess performance metrics.
4. **Data Publishing**: All collected information is stored in a structured format in the `sc-inspector-data` repository.
