# has 10 in-context examples
PROMPT = """\
Given two tables, match column headers if their columns have very similar values. Most columns will not have a match.

Respond with a json list, whose elements are two element lists. The first element is the key of Object 1 and the matching key of Object 2.
For example, if the key 'Dataset size' and 'Number of training examples' are matched, you should return '[['Dataset size', 'Number of training examples']]. If no keys contain the same information, then just output an empty list '[]'

Table 1:
|           | Dataset      | Hugging Face Hub link     | # docs     | # snippets   |
|----------:|:-------------|:--------------------------|:-----------|:-------------|
| 230435736 | ['The Pile'] | ['the_pile_deduplicated'] | ['134M']   | ['673M']     |
| 257378329 | ['ROOTS']    | ['bigscience-data']       | ['598M']   | ['2,171M']   |
| 252917726 | ['LAION']    | ['laion2B-en']            | ['2,322M'] | ['1,351M']   |

Table 2:
|           | Datasets included        | Sizes of the train split                                                                                                                               | Source of Datasets                                                                                                                                                                                                                                                                   | Terms of use for datasets                                           | Pre-processing details                        |
|----------:|:-------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|:----------------------------------------------|
| 230435736 | The Pile: 22 datasets    | N/A                                                                                                                                                    | PubMed Central, ArXiv, GitHub, FreeLaw Project, Stack Exchange, US Patent and Trademark Office, PubMed, Ubuntu IRC, HackerNews, YouTube, PhilPapers, NIH ExPorter, Enron Emails, Books3, Project Gutenberg, OpenSubtitles, English Wikipedia, DM Mathematics, EuroParl, Enron Emails | Terms of Service compliance discussed, with mixed adherence to ToS. | Detailed pre-processing information provided. |
| 257378329 | ROOTS, OSCAR, and others | N/A                                                                                                                                                    | BigScience Catalogue, Masader, GitHub, StackExchange, OSCAR                                                                                                                                                                                                                          | N/A                                                                 | Detailed pre-processing information provided. |
| 252917726 | LAION series: 3 datasets | Train split sizes: 2.32 billion English image-text pairs, 2.26 billion image-text pairs from other languages, 1.27 billion undetected language samples | Common Crawl                                                                                                                                                                                                                                                                         | N/A                                                                 | Detailed pre-processing information provided. |

Response: '[["Dataset", "Datasets included"], ["#docs", "Sizes of the train split"]]'


Table 1:
|           | Technique                             | Dataset                                                               |
|----------:|:--------------------------------------|:----------------------------------------------------------------------|
|   2528492 | ['AlexNet and GoogleNet']             | ['54,305 leaf images in 38 classes from PlantVillage dataset']        |
| 233572242 | ['GoogleNet']                         | ['1383 images in 56 classes']                                         |
| 250627360 | ['Residual CNN, attention mechanism'] | ['95,999 tomato leaf images in 10 classes from PlantVillage dataset'] |

Table 2:
|           | Paper title                                | Abstract content   |
|----------:|:-------------------------------------------|:-------------------|
|   2528492 | Smartphone-Assisted Crop Disease Diagnosis | N/A                |
| 233572242 | Squeeze-and-Excitation MobileNet Model     | Contains abstract  |
| 250627360 | Vision Transformer Enabled CNN             | Contains abstract  |

Response: '[["Application", "Application Domain"], ["Methodology", "Machine Learning-based Approach"]]'


Table 1:
|           | Venue        | Generator                                                                                                                                                                | Discriminator                                                                | Objective Function                                                                                                                |
|----------:|:-------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|
| 248006334 | ["ECCV'22"]  | ['Generator with time-agnostic 3D VQGAN and time-sensitive Trasformer']                                                                                                  | ['Two discriminators: a spatial discriminator and a temporal discriminator'] | ['VQGAN: Adversarial loss, matching loss, reconstruction loss, codebook loss, commit loss, Transformer: Negative log-likelihood'] |
| 254563906 | ["arXiv'22"] | ['Generator consisting of the 3D-VQ Encoder, Bidirectional Transformer and 3D-VQ Decoder']                                                                               | ['StyleGAN-based 3D discriminator']                                          | ['GAN loss, image perceptual loss, LeCam regularization, reconstruction loss, refine loss, and masking loss']                     |
| 237431154 | ["ICCV'21"]  | ['A sub-token fusion enabled Transformer with a soft split and composition method with CNN encoder and decoder']                                                         | ['CNN-based video discriminator']                                            | ['Adversarial loss and reconstruction loss']                                                                                      |
| 252568225 | ["ACMMM'22"] | ['Generator with Encoder, Patch-based deformed Transformer, and decoder']                                                                                                | ['Temporal PatchGAN-based discriminator']                                    | ['Adversarial loss and reconstruction loss on hole and valid pixels']                                                             |
| 251564308 | ["ECCV'22"]  | ['Generator with flow-guided content propagation, spatial and temporal Transformers']                                                                                    | ['Temporal PatchGAN-based discriminator']                                    | ['Adversarial loss and reconstruction loss']                                                                                      |
| 256194123 | ["arXiv'23"] | ['FGT with flow-guided feature integration and flow-guided feature propagation modules']                                                                                 | ['Temporal PatchGAN-based discriminator']                                    | ['Adversarial loss, spatial domain reconstruction loss and amplitude loss']                                                       |
| 236493410 | ["ACMMM'21"] | ['Convolutional transformer with encoder, temporal self-attention module and decoder']                                                                                   | ['Two discriminators: 2D Conv and 3D Conv-based discriminators']             | ['Adversarial loss and pixel-wise L1 loss']                                                                                       |
| 249953817 | ["ICLR'23"]  | ['Generator with VQGAN and Bidirectional window transformer for variable percentage masked tokens prediction']                                                           | ['VQ-GAN discriminator']                                                     | ['Adversarial loss, perceptual loss and reconstruction loss']                                                                     |
| 247839794 | ["CVPR'22"]  | ['A bi-directional RNN architecture having temporal aggregation module in masked encoder and flow features and spatial restoration transformer followed by Conv layers'] | ['Temporal PatchGAN discriminator']                                          | ['Spatial-temporal adversarial loss, L1 loss and perceptual loss']                                                                |

Table 2:
|           | Generator Network                   | Discriminator Network                                          | Objective Function                     | Application and Datasets                                                                                          |
|----------:|:------------------------------------|:---------------------------------------------------------------|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------|
| 248006334 | TATS model                          | N/A                                                            | Yes                                    | Video Generation: UCF-101, Sky Time-lapse, Taichi-HD, AudioSet-Drum, MUGEN                                        |
| 254563906 | MAGVIT                              | N/A                                                            | Minimize cross-entropy                 | Video Generation and Manipulation: UCF-101, BAIR Robot Pushing, Kinetics-600, nuScenes, Objectron, 12M Web Videos |
| 237431154 | FuseFormer model                    | Uses a discriminator network for FuseFormer generator training | N/A                                    | Video Inpainting: YouTube-VOS, DAVIS                                                                              |
| 252568225 | Detailed description provided       | Uses Temporal PatchGAN (T-PatchGAN) for adversarial training   | Yes                                    | Video Inpainting: DAVIS, Youtube-VOS                                                                              |
| 251564308 | N/A                                 | N/A                                                            | Yes                                    | Video Inpainting: Youtube-VOS, DAVIS                                                                              |
| 256194123 | N/A                                 | N/A                                                            | Yes                                    | N/A                                                                                                               |
| 236493410 | CT-D2GAN framework                  | Uses a dual discriminator network in CT-D2GAN framework        | Minimize negative log-likelihood       | Video Anomaly Detection                                                                                           |
| 249953817 | VQ-GAN                              | N/A                                                            | Combination of multiple loss functions | Video Prediction: BAIR robot pushing, KITTI, RoboNet                                                              |
| 247839794 | Recurrent transformer network (RTN) | Uses a discriminator network for adversarial training          | nan                                    | Film Restoration: Synthetic datasets, Real-world old films                                                        |

Response: '[[]]'


Table 1:
|          | Application                         | Methodology                |
|---------:|:------------------------------------|:---------------------------|
| 29873442 | ['Distributed resource management'] | ['Reinforcement learning'] |
|  3930912 | ['Vehicle trajectory prediction']   | ['Reinforcement learning'] |

Table 2:
|          | Machine Learning-based Approach     | Application Domain                             |
|---------:|:------------------------------------|:-----------------------------------------------|
| 29873442 | Deep reinforcement learning         | Vehicle-to-Vehicle (V2V) Communication Systems |
|  3930912 | General machine learning techniques | Vehicle Trajectory Prediction                  |

Response: '[[]]'


Table 1:
|           | Organ     | Model                    | Dataset                                                  | Metrics                    |
|----------:|:----------|:-------------------------|:---------------------------------------------------------|:---------------------------|
|  49862143 | ['Heart'] | ['VAE {{cite:d9bba4c}}'] | ['Multi-centre {{cite:8c8a7e0}}, ACDC {{cite:6e98983}}'] | ['ACC']                    |
| 220042182 | ['Heart'] | ['VAE {{cite:d9bba4c}}'] | ['Biobank {{cite:e8ba0ea}}']                             | ['Balanced ACC, Sen, Spe'] |

Table 2:
|           | Title                                              | Interpretability                  | Accuracy                                                          | Dataset                                                                                                              |
|----------:|:---------------------------------------------------|:----------------------------------|:------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
|  49862143 | Deep Learning of Interpretable Anatomical Features | 3D convolutional generative model | 100% accuracy on testing dataset, 90% on ACDC MICCAI 2017 dataset | Multi-centre cohort: 686 patients with HCM, 679 healthy volunteers; CMR imaging at 1.5-T on Siemens/Philips systems. |
| 220042182 | N/A                                                | Variational autoencoder (VAE)     | N/A                                                               | 73 patients; CMR and 2D echocardiography pre and post 6-months CRT; Multi-slice SA stack used.                       |

Response: '[["Dataset", "Dataset"]]'


Table 1:
|           | Techniques                        | Problem                                                                                                        |
|----------:|:----------------------------------|:---------------------------------------------------------------------------------------------------------------|
| 237347064 | ['Cross-Modal Feature Selection'] | ['Select features that carry complementary information across different modalities for effective fusion.']     |
| 233834400 | ['Evaluation and Validation']     | ['Evaluate feature selection and fusion methods using appropriate metrics, and validate their effectiveness.'] |

Table 2:
|           | Multimodal Fusion                                     | Feature Selection   |
|----------:|:------------------------------------------------------|:--------------------|
| 237347064 | Uses multi-modal information: RGB and flow modalities | N/A                 |
| 233834400 | N/A                                                   | Yes                 |

Response: '[[]]'


Table 1:
|           | Pretraining Corpus Type      | Models                                                                                                                                            |
|----------:|:-----------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| 221640658 | ['Monolingual (Indonesian)'] | ['IndonesianBERT {{cite:171c52e}}']                                                                                                               |
| 211818153 | ['Monolingual (Chinese)']    | ['RoBERTa-tiny-clue {{cite:6a18fcf}}']                                                                                                            |
| 233289621 | ['Multilingual']             | ['IndoBART {{cite:d3f39d6}}']                                                                                                                     |
| 225040574 | ['Multilingual']             | ['mT5 {{cite:4156fab}}']                                                                                                                          |
| 207870323 | ['Multilingual']             | ['mT6 {{cite:ef49e2b}}']                                                                                                                          |
| 233240686 | ['Multilingual']             | ['IndT5 {{cite:f8246e4}}']                                                                                                                        |
| 210919995 | ['Parallel']                 | ['MuRIL {{cite:6e4abd3}}']                                                                                                                        |
|  21709048 | ['Parallel']                 | ['mT6 {{cite:ef49e2b}}, XLM {{cite:39539ef}}, infoXLM {{cite:044a7b2}}, Unicoder {{cite:fbf3b7e}}, ALM {{cite:86b0046}}, XLM-E {{cite:fa83910}}'] |
| 196471198 | ['Parallel']                 | ['mT6 {{cite:ef49e2b}}, XLM-E {{cite:fa83910}}']                                                                                                  |
| 208006253 | ['Parallel']                 | ['XLM-E {{cite:fa83910}}']                                                                                                                        |
| 218973987 | ['Parallel']                 | ['MuRIL {{cite:6e4abd3}}']                                                                                                                        |
| 233210367 | ['Parallel']                 | ['-']                                                                                                                                             |

Table 2:
|           | Language                        | Corpus Size (e.g. tokens, number of sentences)                                                             |
|----------:|:--------------------------------|:-----------------------------------------------------------------------------------------------------------|
| 221640658 | Indonesian                      | 4 billion words, 250 million sentences                                                                     |
| 211818153 | Chinese                         | 100 GB                                                                                                     |
| 233289621 | Indonesian, Javanese, Sundanese | N/A                                                                                                        |
| 225040574 | Not specified                   | N/A                                                                                                        |
| 207870323 | Not specified                   | 3.2 TB compressed, 532 billion tokens (English), 101 billion tokens (Russian), 92 billion tokens (Chinese) |
| 233240686 | Indigenous languages, Spanish   | 1.17 GB, 5.37 million sentences                                                                            |
| 210919995 | Indo-Aryan, Dravidian           | N/A                                                                                                        |
|  21709048 | English, Hindi                  | 1.49 million parallel segments                                                                             |
| 196471198 | Multiple languages              | N/A                                                                                                        |
| 208006253 | Multiple languages              | N/A                                                                                                        |
| 218973987 | 12 South Asian languages        | N/A                                                                                                        |
| 233210367 | English, 11 Indic languages     | 49.7 million sentence pairs                                                                                |

Response: '[[]]'


Table 1:
|           | Language   | Model                                         | Pretraining Corpora             |
|----------:|:-----------|:----------------------------------------------|:--------------------------------|
| 224814107 | ['(da)']   | ['DJSammy/bert-base-danish-uncased_BotXO,ai'] | ['Wikipedia + Web + Subtitles'] |
| 210714061 | ['(de)']   | ['deepset/gbert-base']                        | ['Wikipedia + OSCAR + OPUS']    |
| 207853304 | ['(nl)']   | ['pdelobelle/robbert-v2-dutch-base']          | ['Wikipedia + Books + News']    |
| 232335496 | ['(sv)']   | ['KB/bert-base-swedish-cased']                | ['Wikipedia + Books + News']    |
| 209376347 | ['(fr)']   | ['camembert-base']                            | ['OSCAR']                       |
| 252438849 | ['(it)']   | ['Musixmatch/umberto-commoncrawl-cased-v1']   | ['Wikipedia + OPUS']            |

Table 2:
|           | BERT Model Name   | Training Corpus Used                                      | Language of the Model   |
|----------:|:------------------|:----------------------------------------------------------|:------------------------|
| 224814107 | GBERT             | OSCAR, German Wikipedia, OPUS, German court decisions     | German                  |
| 210714061 | RobBERT           | OSCAR Dutch corpus (6.6 billion words, 126,064,722 lines) | Dutch                   |
| 207853304 | CamemBERT         | OSCAR French corpus                                       | French                  |
| 232335496 | Czert-B           | CsNat, CsWiki, CsNews                                     | Czech                   |
| 209376347 | FinBERT           | Finnish news, online discussion, internet crawl           | Finnish                 |
| 252438849 | N/A               | N/A                                                       | N/A                     |

Response: '[["Language", "Language of the Model"], ["Model", "BERT Model Name"], ["Pretraining Corpora", "Training Corpus Used"]]'


Table 1:
|          | Size (MB)   | Classes   | Domain                     | Task                            |
|---------:|:------------|:----------|:---------------------------|:--------------------------------|
|   368182 | ['1427']    | ['2']     | ['Product reviews']        | ['sentiment classification']    |
|  5034059 | ['43']      | ['2']     | ['Online forum questions'] | ['paraphrase detection']        |
|  3432876 | ['65']      | ['3']     | ['Diverse']                | ['natural language inference']  |
| 91184042 | ['139']     | ['2']     | ['Wikipedia']              | ['paraphrase detection']        |
| 85543290 | ['293']     | ['174']   | ['Web crawl']              | ['discourse marker prediction'] |

Table 2:
|          | Paper Title                                                                           | Paper Abstract                                                                                                                                    | NLU Task                 | Dataset Size                              |
|---------:|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------|:------------------------------------------|
|   368182 | Character-level Convolutional Networks for Text Classification                        | Empirical exploration of character-level ConvNets for text classification with comparisons to traditional and deep learning models.               | N/A                      | Hundreds of thousands to several millions |
|  5034059 | GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding | Introduces GLUE benchmark to evaluate NLU models, emphasizing cross-task generalization and detailed linguistic analysis.                         | Discusses GLUE benchmark | N/A                                       |
|  3432876 | Multi-Genre Natural Language Inference (MultiNLI) corpus                              | Introduces MultiNLI dataset for developing sentence understanding machine learning models.                                                        | Mentions NLU task: NLI   | 433,000 examples                          |
| 91184042 | PAWS: Paraphrase Adversaries from Word Scrambling                                     | Introduces PAWS dataset with high lexical overlap paraphrase and non-paraphrase pairs to improve model distinctions.                              | N/A                      | 108,463 pairs                             |
| 85543290 | Discovery of Discourse Markers for Sentence Representation Learning                   | Proposes a method for automatic discovery of sentence pairs with discourse markers for learning sentence embeddings; datasets publicly available. | N/A                      | N/A                                       |

Response: '[["Size (MB)", "Dataset Size"], ["Task", "NLU Task"]]'


Table 1:
|          | ID      | Dataset           | Type     | #Subject   | S. Rate        | #Activity   | #Sample        | Sensor   |
|---------:|:--------|:------------------|:---------|:-----------|:---------------|:------------|:---------------|:---------|
|  7551351 | ['D06'] | ['WISDM']         | ['ADL']  | ['29']     | ['20 Hz']      | ['6']       | ['1,098,207']  | ['A']    |
| 12228599 | ['D14'] | ['Daphnet Gait']  | ['Gait'] | ['10']     | ['64 Hz']      | ['2']       | ['1,917,887']  | ['A']    |
|  5249903 | ['D19'] | ['Heterogeneous'] | ['ADL']  | ['9']      | ['100-200 Hz'] | ['6']       | ['43,930,257'] | ['A, G'] |

Table 2:
|          | Recognition accuracy                           | Computational cost   | Sensor modality         | Model architecture                                                                           | Implementation platform   | Dataset used                                | Evaluation metric                                                   | Methodology                                                                                                                                                                                           |
|---------:|:-----------------------------------------------|:---------------------|:------------------------|:---------------------------------------------------------------------------------------------|:--------------------------|:--------------------------------------------|:--------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  7551351 | Improved recognition accuracy with deep models | O(L x D)             | triaxial accelerometers | Deep Belief Networks with Gaussian-binary RBM and multi-layer binary-binary RBMs             | N/A                       | Uses WISDM, Daphnet Freezing of Gait, Skoda | Accuracy                                                            | Uses deep learning and triaxial accelerometers for activity recognition with unsupervised pre-training and supervised fine-tuning. Employs spectrogram representation and HMMs for temporal patterns. |
| 12228599 | N/A                                            | N/A                  | N/A                     | Deep Neural Networks, Convolutional Neural Networks, and Recurrent Neural Networks with LSTM | N/A                       | Uses Opportunity, PAMAP2, Daphnet Gait      | Mean F1-score                                                       | Evaluates deep learning for HAR using wearable sensors. Compares DNNs, CNNs, and RNNs across datasets with detailed analysis of hyperparameters using fANOVA.                                         |
|  5249903 | Recognition accuracy: 0.9421 0.032                                                | N/A                  | motion sensors          | N/A                                                                                          | Yes                       | N/A                                         | Accuracy, Macro F1 Score, Micro F1 Score, Mean Absolute Error (MAE) | N/A                                                                                                                                                                                                   |

Response: '[["Dataset", "Dataset used"]]'\
"""