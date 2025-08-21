# Financial Time Series Forecasting – Checklist

## Classical Econometrics & Volatility
- [ ] [AR / MA / ARIMA / SARIMA / ARIMAX](https://otexts.com/fpp3/arima.html)
- [ ] [ETS / Holt–Winters Exponential Smoothing](https://robjhyndman.com/expsmooth/) • [State-space ETS](https://otexts.com/fpp2/ets.html)
- [ ] [Theta Method (incl. Optimised Theta)](https://arxiv.org/pdf/1503.03529)
- [ ] [ARFIMA (long memory)](https://www.sciencedirect.com/science/article/abs/pii/S0304405X02001383)
- [ ] [VAR / VECM (cointegration)](https://www.econometrics-with-r.org/11-1-vector-autoregressions.html)
- [ ] [Johansen Cointegration Test](https://en.wikipedia.org/wiki/Johansen_test)
- [ ] [State-Space Models & Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter)
- [ ] [Bayesian Structural Time Series (BSTS)](https://cran.r-project.org/web/packages/bsts/vignettes/bsts.html)
- [ ] [Markov-/Regime-Switching (MS-AR, HMM)](https://en.wikipedia.org/wiki/Markov_switching_model)
- [ ] [GARCH family (GARCH/EGARCH/GJR/DCC)](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity)
- [ ] [Stochastic Volatility](https://en.wikipedia.org/wiki/Stochastic_volatility)
- [ ] [HAR-RV (Realized Volatility)](https://academic.oup.com/rfs/article-abstract/16/1/67/1587394)
- [ ] [MIDAS (mixed-frequency regressions)](https://en.wikipedia.org/wiki/MIDAS_regression_model)
- [ ] [Dynamic Factor Models & FAVAR](https://www.jstor.org/stable/3695610)

## Dependence & Risk
- [ ] [Copulas (static & conditional)](https://en.wikipedia.org/wiki/Copula_(probability_theory))
- [ ] [Vine Copulas](https://en.wikipedia.org/wiki/Vine_copula)

## Feature-Engineered Machine Learning
- [ ] [Regularized Linear (Ridge/Lasso/Elastic Net)](https://scikit-learn.org/stable/modules/linear_model.html)
- [ ] [Quantile Regression](https://en.wikipedia.org/wiki/Quantile_regression)
- [ ] [Support Vector Regression (SVR)](https://scikit-learn.org/stable/modules/svm.html#regression)
- [ ] [Tree Ensembles: RF / XGBoost / LightGBM / CatBoost](https://xgboost.readthedocs.io/en/stable/)

## Bayesian & Non-parametric
- [ ] [Gaussian Processes for TS](https://gaussianprocess.org/gpml/chapters/RW.pdf)

## Deep Learning (Non-Transformer)
- [ ] [LSTM / GRU (variants, attention)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [ ] [Temporal Convolutional Networks (TCN)](https://arxiv.org/abs/1803.01271)
- [ ] [WaveNet-style 1D CNNs](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
- [ ] [N-BEATS / N-HiTS](https://arxiv.org/abs/1905.10437)

## Transformer-Based & Modern Sequence Models
- [ ] [Temporal Fusion Transformer (TFT)](https://arxiv.org/abs/1912.09363)
- [ ] [Informer](https://arxiv.org/abs/2012.07436) • [Autoformer](https://arxiv.org/abs/2106.13008) • [FEDformer](https://arxiv.org/abs/2201.12740)
- [ ] [PatchTST](https://arxiv.org/abs/2211.14730) • [TimesNet](https://arxiv.org/abs/2210.02186)
- [ ] [ETSformer](https://arxiv.org/abs/2202.01381) • [iTransformer](https://arxiv.org/abs/2310.06625)
- [ ] [DLinear / NLinear (strong linear baselines)](https://arxiv.org/abs/2205.13504)
- [ ] [TSMixer / All-MLP forecasters](https://arxiv.org/abs/2303.06053)
- [ ] [Structured State-Space Models: S4](https://arxiv.org/abs/2111.00396) • [Mamba (selective SSM)](https://arxiv.org/abs/2312.00752)

## Foundation Models & LLM-Style “Next-Token” TS
- [ ] [TimesFM (Google)](https://arxiv.org/abs/2403.07815)
- [ ] [Chronos (Amazon)](https://arxiv.org/abs/2403.07815) • [Repo](https://github.com/amazon-science/chronos-forecasting)
- [ ] [TimeGPT (Nixtla)](https://nixtla.github.io/)
- [ ] [Lag-Llama](https://arxiv.org/abs/2310.08278)
- [ ] [Moirai / Moirai-MoE (Salesforce)](https://arxiv.org/abs/2310.06625)
- [ ] [Time-LLM](https://arxiv.org/abs/2310.10688) • [LLM4TS](https://arxiv.org/abs/2310.00752)
- [ ] [GPT-style TS Forecasters (e.g., TEMPO / Timer)](https://arxiv.org/abs/2402.02368)

## Probabilistic & Generative TS
- [ ] [DeepAR](https://arxiv.org/abs/1704.04110) • [DeepVAR](https://ts.gluon.ai/api/gluonts/gluonts.model.deepvar.html) • [DeepState](https://arxiv.org/abs/1906.01949)
- [ ] [Normalizing Flows (RealNVP)](https://arxiv.org/abs/1605.08803)
- [ ] [Diffusion/Score-based TS: TimeGrad](https://arxiv.org/abs/2101.06182) • [CSDI (imputation)](https://arxiv.org/abs/2107.03502)
- [ ] [Variational RNNs (VRNN/SRNN)](https://arxiv.org/abs/1506.02216)
- [ ] [GAN-based (TimeGAN)](https://arxiv.org/abs/1907.05321)

## Continuous-Time / Irregular Sampling
- [ ] [Neural ODEs](https://arxiv.org/abs/1806.07366) • [Neural CDEs](https://arxiv.org/abs/2005.08926) • [Neural SDEs](https://arxiv.org/abs/1906.02355)

## Operator-Theoretic / Dynamical Systems
- [ ] [Koopman Operator Models (Deep/Prob.)](https://arxiv.org/abs/2211.07561)
- [ ] [Dynamic Mode Decomposition (DMD)](https://arxiv.org/abs/2312.00137)

## Reservoir Computing
- [ ] [Echo State Networks (ESN)](https://en.wikipedia.org/wiki/Echo_state_network)

## Hierarchical / Grouped Forecasting
- [ ] [Forecast Reconciliation (MinT & variants)](https://robjhyndman.com/papers/mint.pdf)

## Multivariate Econometric Extensions
- [ ] [TVP-VAR (time-varying parameter VAR)](https://www.richmondfed.org/-/media/richmondfedorg/publications/research/economic_quarterly/2015/q4/pdf/lubik.pdf)
- [ ] [Bayesian VARs / Large VARs w/ Stochastic Volatility](https://onlinelibrary.wiley.com/doi/full/10.1002/jae.2936)

## Change-Point / Regime Detection
- [ ] [BOCPD (Bayesian Online CPD)](https://arxiv.org/abs/0710.3742)
- [ ] [PELT & Offline CPD (ruptures library)](https://centre-borelli.github.io/ruptures-docs/) • [Survey](https://www.sciencedirect.com/science/article/abs/pii/S0165168419303494)

## Graph-Based & Multimodal (Equities)
- [ ] [Relational/Temporal GNNs (Stock Ranking)](https://arxiv.org/abs/1909.10660)
- [ ] [HATS: Hierarchical Graph Attention](https://arxiv.org/abs/1908.07999)
- [ ] [FinBERT Sentiment + Price Fusion](https://github.com/ProsusAI/finBERT)

## High-Frequency & Microstructure
- [ ] [Hawkes / Self-Exciting Point Processes](https://en.wikipedia.org/wiki/Hawkes_process)
- [ ] [DeepLOB (CNN/LSTM/TCN on LOB)](https://arxiv.org/abs/1808.03668)
- [ ] [Transformer-based LOB (e.g., TLOB / LOB Transformers)](https://arxiv.org/abs/2003.00130)

## Reinforcement Learning (Trading/Execution/PM)
- [ ] [Deep RL for Portfolio Management (Jiang et al.)](https://arxiv.org/abs/1706.10059)
- [ ] [FinRL (PPO, DDPG/TD3, SAC, etc.)](https://github.com/AI4Finance-Foundation/FinRL)

## Causal & Regime-Aware
- [ ] [Granger Causality](https://en.wikipedia.org/wiki/Granger_causality)
- [ ] [CausalImpact (BSTS intervention analysis)](https://google.github.io/CausalImpact/CausalImpact.html)
- [ ] [Markov-Switching (Hamilton-style)](https://en.wikipedia.org/wiki/Markov_switching_model)

## Other Workhorses
- [ ] [TBATS (multiple/complex seasonality)](https://otexts.com/fpp3/tbats.html)
- [ ] [Prophet](https://facebook.github.io/prophet/)
- [ ] [TRMF (Temporal Reg. Matrix Factorization)](https://arxiv.org/abs/1704.04110)
- [ ] [Wavelet-Based Time-Frequency Methods](https://en.wikipedia.org/wiki/Wavelet_transform)
