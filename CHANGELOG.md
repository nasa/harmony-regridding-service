## v0.0.2
### 2023-10-17

No user facing changes.  Migrates sds-varinfo to the public facing
earthdata-varinfo package.

## v0.0.1
### 2023-03-28

Initial minimum viable product version of the harmony-regridding-service.  The
service now can be added to harmony to return a resampled granule. The service
is limited to resampling geographic data only. If an input granule has multiple
grids defined, the data from all grids will be resampled to the target grid.

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
