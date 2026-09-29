<p align="center">
  <img src="assets/weathergenerator_logo.png" alt="WeatherGenerator" width="400px">
</p>

<div align="center">
  <h1>The WeatherGenerator <br> Machine Learning Earth System Model</h1>
</div>

The WeatherGenerator project is developing a machine learning-based Earth system model. 
It will be trained on a wide range of datasets, including reanalyses, forecast data and observations, to provide a robust and versatile model for the dynamics.
Through this, it can be used for a wide-range of applications. General updates are shared on the project website: [weathergenerator.eu](https://weathergenerator.eu/) 

More details coming soon. Please open an issue if you are interested in using the model.

<hr>

<p align="center">
  <img src="assets/weathergenerator_partner.png" alt="Partners" width="1000px">
</p>

# How to use the WeatherGenerator project

The model is currently being developed by the WeatherGenerator Consortium. If you want to
engage, you are encouraged to contact us first by opening an issue on Github.

## Multiple HEALPix encoders

Encoder architecture settings (`ae_*`) belong under `encoders.<name>`, not at the
top level. Each encoder has independent weights, input resolution, and local/global
architecture. Physical streams declare their encoder memberships; `[]` makes a
stream output-only. One stream may feed several encoders without duplicating its
physical targets or loss weight.

Shape/routing fragment below. Complete encoder blocks also need the remaining
`ae_*` settings shown in [the default configuration](config/default_config.yml);
stream definitions retain their existing reader, embedding, and readout settings.

```yaml
encoders:
  default:
    healpix_level: 5
    ae_local_dim_embed: 1024
    ae_global_dim_embed: 2048
    ae_local_num_queries: 1
  regional:
    healpix_level: 7
    ae_local_dim_embed: 512
    ae_global_dim_embed: 2048
    ae_local_num_queries: 1
fe_healpix_level: 5
fe_dim_embed: 2048
fe_num_queries: 1
streams:
  ERA5:
    encoders: [default]
  LOCAL:
    encoders: [default, regional]
```

Every encoder must match `fe_dim_embed` and `fe_num_queries` at its output.
Class/register-token layout and dtype settings remain shared. There are no output
adapters or learned fusion weights.

At the forecast-input boundary, fine NESTED children are **summed** into each
coarse parent; coarse parents are broadcast to their fine children. Aligned encoder
outputs are added once, with auxiliary tokens added separately. Forecast steps then
advance only that shared state; initial encoder outputs are not added again.

Coverage means occupied input cells before training masks, excluding spoofed or
missing observations. Uncovered cells contribute exactly zero, including auxiliary
tokens for an entirely absent branch. Covered but masked cells retain query/global
processing. This is not sparse computation: level 7 still has 196,608 dense cells.

This is a configuration and checkpoint format break. Stream-level `healpix_level`
and root-level `ae_*` settings are no longer accepted. Checkpoints must contain
`encoders.<name>.*` weights for every configured branch; legacy `encoder.*` keys
and missing branch weights are rejected rather than silently initialized.

# Development guidelines

The [main branch](https://github.com/ecmwf/WeatherGenerator/tree/main) is the most stable version. If you are running experiments, you should use this branch.

The [develop branch](https://github.com/ecmwf/WeatherGenerator/tree/develop) has the latest
features. However, it is currently evolving at a fast pace. It should not be expected to have stable code or weight interfaces, or to be backward compatible.

# Copyright and License

This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.
