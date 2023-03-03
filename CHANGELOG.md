## v0.0.0
### 2023-03-03

This initial version of the Harmony Regridding service sets up the core
infrastructure required for a backend service. It contains a RegriddingServiceAdapter
class that can be invoked within a Harmony Kubernetes cluster, and will return
the input granule without any transformation. The `RegriddingServiceAdapter` performs
basic message validation, to ensure the message defines a valid target grid,
and that it does not specify a target CRS or interpolation method that is
incompatible with the service. The service itself will be implemented in future
tickets.
