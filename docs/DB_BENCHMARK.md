let's make a thorough plan for benchmarking the database performance.

You can inspect pulumi providers' source code at /tmp/pulumi-*.

See "Public Cloud PostgreSQL Benchmarking Roadmap v0.1.pdf" for the generic concept.

Our initial plan is to make our sc-inspector infra (with the benchmarks/tasks running in the sc-inspector-data repo) capable of
starting infra elements for evaluating the database performance.
We want to start with postgres and may add another databases later, so we have to come up with a generic approach, which supports this.
We want to support single-instance benchmarks (the benchmark suite runs on the same machine as the database) and multi-instance benchmarks (the benchmark suite runs on a separate machine as the database).
We want to add this capability to the current sc-inspector infra in the least invasive way possible.
For the single-instance benchmark, we could add the benchmark as a @tasks.py task (as start_with_instance=True, so it doesn't trigger already
benchmarked instances).
For the multi-instance benchmark, we could add the capability to start a companion machine for the benchmark, which runs only the benchmark suite and access the database via the network.
Questions and problems to be solved here:
1. how to handle lifecycle? @tasks.py starts a lot of tasks, we need the companion machine only for the interval until the DB benchmark is running.
2. we have to communicate the database credentials to the benchmark suite, so it can connect to the database.
3. placement is very important, we need to place the benchmark suite close to the database, so the network latency is minimized.
4. instance type selection is very important, we need to select the instance type based on the database size and the benchmark suite requirements.
5. handling errors is very important, we need to handle errors gracefully and not let the benchmark suite fail the entire benchmark. We must not leave any orphaned resources behind, as they cost a lot of money (see @cleanup.yml in sc-inspector-data repo).

Apart from these, we need to benchmark cloud-hosted databases (DBaaS). The idea here is similar, but instead of creating two instances, like
in the multi-instance benchmark, we only create one instance for the benchmark suite, and a hosted DB, like azure-native.azurearcdata.PostgresInstance.
We'll have to implement hosted/managed DBs for different providers (AWS, GCP, OVH, vultr, alicloud, upcloud), so keep this in mind and learn about the providers' APIs/pulumi provider SDKs and bake the knowledge into the sc-inspector infra.

For the MVP, we have to provide:
- single instance benchmark for postgres
- multi instance benchmark for postgres
- single instance benchmark (benchmark suite running on a machine in the same region as the hosted DB, or otherwise close to it) for cloud-hosted postgres

We can use the sc-images repo to build any images we need (arm/amd64) for running the server and the benchmark suite.
The idea is to compile multiple test cases, like OLTP S, M, L, XL (to better utilize the hardware, for example running the S test case on a huge machine with TBs of RAM and hundreds of cores is not relevant, but running the XL will be), and then run the benchmark suite for each of them.
We'll need to add a ladder, so a small instance won't run the XL test case, only the ones which it can handle. We have to auto-determine this,
based on some metrics (like available RAM, CPU cores, disk etc.). An instance must run all tiers it can handle (up from the smallest to the biggest it can support).
We have to provide a central place to describe these test cases and their scaling properties.
The plan is to have a templated or auto-tuned postgres image, which adapts to the machine it's running on.

We'll need to have detailed metrics where available (for example when running on VMs, I'm not sure if we can gather the metrics from the managed DBs, but if we can, that's a plus).
It would be great to store these metrics next to the benchmark results, so we can analyze them later.

Turning to that topic: we could host the single and multi-instance benchmarks in the same sc-inspector-data repo (with the server's vendor/instance_type as a key), also registering the companion machine's vendor/instance_type if there was any.

But we'll have to design a way to store the managed DBs' metrics, similar to the single and multi-instance benchmarks.
Look around the internet and the pulumi providers' source code for inspiration on how to properly name these. The vendor structure is clear (aws, azure, gcp, etc.), but the instance_type is not, as there are a lof ot managed DB types and even though we concentrate on single instance (no multi-instance, replication, multi-region etc) managed DBs, in the future, we might want to support multi-instance, replicated etc, or those managed DBs which have a postgres wire protocol, but don't actually run vanilla postgres, like AWS serverless postgres, or dsql etc.



