# BTC_5s_pred_strategy

Prediction of 5-second forward price on 15-minute BTC up/down Polymarket contracts.

## Repo pieces

- `paper_lstm_model.py`: paper-style feature prep and LSTM scaffolding.
- `strategy_backtest.py`: fee-aware backtest that writes blotter, ledger, and site artifacts.
- `index.html`: GitHub Pages-friendly strategy site.

## Refresh the site data

```bash
python3 strategy_backtest.py
```
