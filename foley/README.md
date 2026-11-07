# OnomaFoley

短文テキストをオノマトペの詩に変換し、OnomaDict 16軸ベクトルと動的モジュレーション・プロフィール (DMP) に基づいて、
映画のフォーリー演者のように効果音を即時合成する Web Audio アプリです。

**Demo:** https://richiowaki3.github.io/OnomaFoley/foley/

## エンジン
- 🔊 TTS 読み上げ — Web Speech API（日本語音声がある環境で動作）
- 🗣 Engine A — LF声帯パルス + 3並列フォルマント (F1/F2/F3, Locusコサイン滑走) + 口唇放射
- 💥 Engine B — Hertz接触インパルス sin^1.5 + モーダル共鳴バンク（木・膜・金属・液体）+ Sub-Kick + FXチェーン
- 🎬 Cross-Foley — `(1-V)·Phys + V·Voice + 2V(1-V)·0.5·[Impact→Formant + Voice→Modal]`

## 実行
依存なしの単一ファイル `index.html`。ブラウザで開くか、静的サーバで配信してください。

```
python -m http.server 8000
```
