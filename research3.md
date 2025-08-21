# Financial Time Series Forecasting – Methods & Hedge Fund Relevance

| Category | Method | Reference | Relevance in Hedge/Commodity/Futures Forecasting |
|----------|--------|-----------|--------------------------------------------------|
| **Classical Econometrics** | [ARIMA / SARIMA / ARIMAX](https://otexts.com/fpp3/arima.html) | Core baseline for univariate price forecasting; used for benchmarking |
| | [ETS / Holt–Winters](https://robjhyndman.com/expsmooth/) | Seasonal patterns (e.g., commodities, demand); less robust in finance |
| | [Theta Method](https://arxiv.org/pdf/1503.03529) | Retail/energy demand; limited hedge fund use |
| | [ARFIMA](https://www.sciencedirect.com/science/article/abs/pii/S0304405X02001383) | Long-memory volatility/return persistence modeling |
| | [VAR / VECM](https://www.econometrics-with-r.org/11-1-vector-autoregressions.html) | Multi-asset & macro linkages (FX, equity sectors, rates) |
| | [Johansen Cointegration](https://en.wikipedia.org/wiki/Johansen_test) | Pairs/stat arb, relative value, spread trading |
| | [State-Space & Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) | Real-time signal extraction, trend filtering, latent factors |
| | [BSTS](https://cran.r-project.org/web/packages/bsts/vignettes/bsts.html) | Event impact analysis, causal inference, macro events |
| | [Markov-Switching / HMM](https://en.wikipedia.org/wiki/Markov_switching_model) | Regime detection (bull/bear, volatility regimes) |
| | [GARCH family](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity) | **Core**: volatility forecasting, VaR, option pricing |
| | [Stochastic Volatility](https://en.wikipedia.org/wiki/Stochastic_volatility) | Options, volatility surface modeling |
| | [HAR-RV](https://academic.oup.com/rfs/article-abstract/16/1/67/1587394) | Volatility forecasting with realized measures (futures/FX) |
| | [MIDAS](https://en.wikipedia.org/wiki/MIDAS_regression_model) | Mixed-frequency (daily + macro monthly), commodities |
| | [Dynamic Factor Models & FAVAR](https://www.jstor.org/stable/3695610) | Macro-financial linkages, yield curve, global factors |
| **Dependence & Risk** | [Copulas](https://en.wikipedia.org/wiki/Copula_(probability_theory)) | Cross-asset dependence, tail risk, structured products |
| | [Vine Copulas](https://en.wikipedia.org/wiki/Vine_copula) | High-dimensional portfolios (credit, commodities baskets) |
| **Machine Learning** | [Regularized Regression](https://scikit-learn.org/stable/modules/linear_model.html) | Factor models, shrinkage in large universes |
| | [Quantile Regression](https://en.wikipedia.org/wiki/Quantile_regression) | Risk forecasting, downside tail models |
| | [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) | Non-linear asset returns, option Greeks approximation |
| | [Tree Ensembles](https://xgboost.readthedocs.io/en/stable/) | **Core**: tabular signals, feature interactions, execution |
| **Bayesian/Nonparametric** | [Gaussian Processes](https://gaussianprocess.org/gpml/chapters/RW.pdf) | Small data, uncertainty estimation, niche signal models |
| **Deep Learning** | [LSTM / GRU](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Popular in hedge funds; intraday & swing signal models |
| | [TCN](https://arxiv.org/abs/1803.01271) | Parallelizable alternative to RNNs; high-freq prediction |
| | [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) | Tick data, order flow modeling |
| | [N-BEATS / N-HiTS](https://arxiv.org/abs/1905.10437) | Strong general-purpose forecasters; benchmark in funds |
| **Transformers & Modern Seq Models** | [TFT](https://arxiv.org/abs/1912.09363) | **High relevance**: interpretable, multi-horizon with exogenous vars |
| | [Informer](https://arxiv.org/abs/2012.07436) / [Autoformer](https://arxiv.org/abs/2106.13008) / [FEDformer](https://arxiv.org/abs/2201.12740) | Long-horizon forecasting, macro/futures term-structure |
| | [PatchTST](https://arxiv.org/abs/2211.14730) | State-of-the-art for multivariate TS forecasting |
| | [TimesNet](https://arxiv.org/abs/2210.02186) | Captures 2D temporal patterns; experimental in finance |
| | [ETSformer](https://arxiv.org/abs/2202.01381) / [iTransformer](https://arxiv.org/abs/2310.06625) | Advanced SOTA; hedge R&D |
| | [DLinear / NLinear](https://arxiv.org/abs/2205.13504) | Strong simple baselines; used in finance research |
| | [TSMixer](https://arxiv.org/abs/2303.06053) | All-MLP forecaster; light & fast for large universes |
| | [S4](https://arxiv.org/abs/2111.00396) / [Mamba](https://arxiv.org/abs/2312.00752) | Long-sequence models; **frontier** for high-frequency data |
| **Foundation Models (Next-Token TS)** | [TimesFM](https://arxiv.org/abs/2403.07815) | **Frontier**: cross-domain TS foundation model |
| | [Chronos](https://arxiv.org/abs/2403.07815) | Zero-/few-shot TS forecasting; finance-ready |
| | [TimeGPT](https://nixtla.github.io/) | Commercial universal forecaster |
| | [Lag-Llama](https://arxiv.org/abs/2310.08278) | Decoder-only TS FM; finance R&D |
| | [Moirai](https://arxiv.org/abs/2310.06625) | TS mixture-of-experts foundation model |
| | [Time-LLM](https://arxiv.org/abs/2310.10688) / [LLM4TS](https://arxiv.org/abs/2310.00752) | Adapting LLMs for TS; very experimental |
| **Probabilistic & Generative** | [DeepAR](https://arxiv.org/abs/1704.04110) / [DeepVAR](https://ts.gluon.ai/api/gluonts/gluonts.model.deepvar.html) / [DeepState](https://arxiv.org/abs/1906.01949) | Probabilistic returns & risk; scenario generation |
| | [Normalizing Flows (RealNVP)](https://arxiv.org/abs/1605.08803) | Flexible risk distributions |
| | [Diffusion Models (TimeGrad, CSDI)](https://arxiv.org/abs/2101.06182) | Scenario stress testing; experimental in hedge funds |
| | [VRNN / SRNN](https://arxiv.org/abs/1506.02216) | Generative sequence models |
| | [TimeGAN](https://arxiv.org/abs/1907.05321) | Synthetic financial data; risk backtesting |
| **Continuous-Time** | [Neural ODE](https://arxiv.org/abs/1806.07366) / [Neural CDE](https://arxiv.org/abs/2005.08926) / [Neural SDE](https://arxiv.org/abs/1906.02355) | Option surfaces, continuous trading dynamics |
| **Operator-Theoretic** | [Koopman Models](https://arxiv.org/abs/2211.07561) | Nonlinear dynamics, volatility surfaces |
| | [Dynamic Mode Decomposition](https://arxiv.org/abs/2312.00137) | Market regime discovery |
| **Reservoir Computing** | [Echo State Networks](https://en.wikipedia.org/wiki/Echo_state_network) | Low-cost forecasting; niche HFT research |
| **Hierarchical Forecasting** | [Forecast Reconciliation (MinT)](https://robjhyndman.com/papers/mint.pdf) | Sector → index aggregation in equities/commodities |
| **Multivariate Econometrics** | [TVP-VAR](https://www.richmondfed.org/-/media/richmondfedorg/publications/research/economic_quarterly/2015/q4/pdf/lubik.pdf) / [BVAR](https://onlinelibrary.wiley.com/doi/full/10.1002/jae.2936) | Macro forecasting, multi-asset stress scenarios |
| **Change-Point Detection** | [BOCPD](https://arxiv.org/abs/0710.3742) | Online regime shift detection |
| | [PELT / Ruptures](https://centre-borelli.github.io/ruptures-docs/) | Offline change-points in financial series |
| **Graph & Multimodal** | [Relational GNNs](https://arxiv.org/abs/1909.10660) | Sector/stock dependency modeling |
| | [HATS](https://arxiv.org/abs/1908.07999) | Hierarchical stock relations |
| | [FinBERT + TS Fusion](https://github.com/ProsusAI/finBERT) | Sentiment + price signal integration |
| **High-Frequency / Microstructure** | [Hawkes Processes](https://en.wikipedia.org/wiki/Hawkes_process) | **Core in HFT**: trade arrival, order flow modeling |
| | [Neural Hawkes](https://arxiv.org/abs/1612.09328) | Learning event dynamics |
| | [DeepLOB](https://arxiv.org/abs/1808.03668) | Limit Order Book forecasting |
| | [LOB Transformers](https://arxiv.org/abs/2003.00130) | Cutting-edge for HFT execution |
| **Reinforcement Learning** | [Deep RL for Portfolio Mgmt](https://arxiv.org/abs/1706.10059) | Allocation, dynamic hedging, policy optimization |
| | [FinRL Library](https://github.com/AI4Finance-Foundation/FinRL) | Open-source DRL trading research |
| **Causal & Regime-Aware** | [Granger Causality](https://en.wikipedia.org/wiki/Granger_causality) | Macro & lead-lag relationships |
| | [CausalImpact](https://google.github.io/CausalImpact/CausalImpact.html) | Event-driven trading (earnings, policy) |
| | [Markov-Switching](https://en.wikipedia.org/wiki/Markov_switching_model) | Regime detection for risk/vol |
| **Other Workhorses** | [TBATS](https://otexts.com/fpp3/tbats.html) | Long seasonality; limited hedge use |
| | [Prophet](https://facebook.github.io/prophet/) | Simple forecasting; rare in hedge funds |
| | [TRMF](https://arxiv.org/abs/1704.04110) | Matrix factorization; retail/energy > finance |
| | [Wavelets](https://en.wikipedia.org/wiki/Wavelet_transform) | Preprocessing + hybrid methods |
