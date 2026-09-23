# OnomaFoley

**声モデル（Vocal Tract Formants）× 物理衝撃モデル（Physical Impact Modal Resonators）のハイブリッド交差点合成 ＆ TTS音響解析パイプライン**

---

## 概要

`OnomaFoley` は、オノマトペ・擬音語・擬態語を対象として、**「人間の発声モデル（声帯振動・声道フォルマント）」** と **「物理世界の衝突・摩擦・流体モデル（剛性・減衰・重低音過渡衝撃・モーダル共鳴）」** をミリ秒単位で交差（クロスモーダル合成）させ、プロシージャルに物理効果音（Foley SFX）を自動生成・解析する音響工学プラットフォームです。

多言語オノマトペ多次元ベクトル辞書 [OnomaDict](https://github.com/richiowaki3/OnomaDict)（JP: 2,061語 / KR: 5,050語 / AF: 245語）の音響バックエンドとして機能し、TTS（音声合成）音声からの物理・音響特徴量自動抽出、ならびに外部WAVに依存しない数学的・物理的DSP合成エンジンを提供します。

---

## 主要コンポーネント

### 1. デュアル・エンジン・ハイブリッド交差点シンセサイザー (`modular_dsp`)
従来の音声合成（人間の声に偏る）やサンプラー（物理打撃に偏る）の限界を突破するため、2つのエンジンを音素構造に応じて動的にルーティング・モーフィングします。

```
                ┌─── [Engine A: 人間の声モデル] ──────┐
                │   Glottal Pulse Source              │
                │   + Vocal Tract Formants (F1, F2)   │
[入力オノマトペ] ─┤                                     ├─► [モーフィング] ─► [合成音]
                │                                     │     (Vocalness)
                └─── [Engine B: 物理衝撃モデル] ──────┘
                    Shockwave Transient (Sub-Kick)
                    + Modal Resonators (Wood/Metal/Glass)
                    + Vocal Tract Bypass
```

- **Pattern A (重打撃・爆発音 / 例: ドカン, ズシン)**: 声道をバイパスし、Sub-Kick（超低域衝撃波 40-80Hz）＋過渡スパイク＋モーダル共鳴を発動。
- **Pattern B (硬質打撃・金属・ガラス / 例: カツン, パリン)**: 声道をバイパスし、高域通過＋ビットクラッシャー＋急峻な減衰共鳴。
- **Pattern C (流体・摩擦・環境 / 例: サラサラ, シュー)**: ホワイトノイズ／ピンクノイズ＋バンドパス／フェイザー。
- **Pattern D (発声・生体・状態 / 例: オギャー, アハハ)**: 声帯パルス励起＋声道フォルマントフィルター。

### 2. TTS音響解析によるフィルター・特徴量抽出パイプライン (`onomato-audio-analyzer`)
TTS読み上げ音声および効果音実音から、以下の物理・音響特徴量を自動解析・抽出します：
- **剛性（Stiffness Estimate）**: 高域スペクトル成分の立ち上がりとエネルギー比率
- **減衰比（Damping Ratio）**: 音響エンベロープの減衰時定数
- **スペクトル重心（Spectral Centroid）**: 音色の明るさ・硬さの物理指標
- **低域過渡強度（Sub-Kick / Crest Factor）**: 瞬間的な衝撃波の尖鋭度
- **母音フォルマント周波数（F1, F2）**: 声の調音点・共鳴構造

### 3. 音響物理拡張辞書 (`data/onomatopoeia_dictionary_enriched.json`)
7,356語（日本語・韓国語・アフリカ語）のオノマトペ辞書に対し、理論的Laban Effortベクトルと実測音響特徴量（`measured_acoustic`, `measured_effort`, `effort_delta`）を完全統合した拡張データセットです。

---

## クイックスタート

### 1. 依存ライブラリのインストール
```bash
pip install numpy scipy matplotlib streamlit
```

### 2. デュアル・エンジン・デモの起動 (Web UI)
ブラウザ上で声モデル、物理衝撃モデル、ハイブリッド交差点合成の波形・スペクトログラムをリアルタイムに比較・試聴できます。

```bash
# Windows バッチで起動
launch_dual_engine_demo.bat

# またはコマンドラインから起動
streamlit run dual_engine_demo.py
```

### 3. 単語の自動物理音響解析
```bash
python auto_physical_sound_analyzer.py --word ドカン
```

---

## 大容量音声データ（WAV/MP3）の取り扱い方針

Gitリポジトリの肥大化を防ぎ、常にクリーンな状態を維持するため、**読み上げ音声や生録音マスター音源（数GB級のWAV/MP3）はGit追跡の対象外（`.gitignore`）** としています。

- 本リポジトリには、**解析済みの軽量メタデータ（JSON / CSV）および音声合成・解析プログラムコード** のみを完全収録しています。
- 音声データセットの全生データは、必要に応じて外部クラウドストレージ（Google Drive / Hugging Face Datasets 等）経由でダウンロード可能な構成をとっています。

---

## 関連プロジェクト

- **[OnomaDict](https://github.com/richiowaki3/OnomaDict)**: 多言語オノマトペ・多次元ベクトル辞書（本家マスターリポジトリ）
