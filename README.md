# NFT-Sales-Anomaly-Detection
This project performs anomaly detection on NFT sales data to identify unusual patterns or outliers that could indicate fraud, market manipulation, or other irregular activities. The analysis combines classical machine learning (Isolation Forest) and deep learning (Autoencoder) techniques to robustly detect anomalies based on multiple transactional and behavioral features.

Dataset
The dataset (nft_sales.csv) contains NFT sales transaction records with key features including:
Sales: Total sales volume (in USD)
Txns: Number of transactions
Buyers: Number of unique buyers
Owners: Number of unique owners

Data Preprocessing
Cleaned the Sales column by removing dollar signs and commas, converting it to numeric.
Ensured Txns, Buyers, and Owners are numeric types.
Handled missing values by imputing mean values in numeric columns.
Engineered additional features to better capture transactional behavior:
Average Transaction Value = Sales / Txns
Transaction Frequency = Txns / Buyers
Owner-to-Buyer Ratio = Owners / Buyers
Standardized features using StandardScaler to normalize their distributions.

Anomaly Detection Methods
Isolation Forest:
An unsupervised ensemble method effective for high-dimensional data.
Trained on normalized features with a contamination rate of 5% to detect anomalies.
Labels anomalies as -1 and normal points as 1.
Captures outliers based on isolation principles — points that are easier to isolate in feature space are flagged as anomalies.

Autoencoder:
A neural network designed to reconstruct its input; reconstruction errors indicate anomalies.

Architecture:
Encoder: Two Dense layers (32, 16 neurons) with ReLU activations.
Decoder: Two Dense layers (32 neurons, then input dimension) with ReLU and Sigmoid activations.
Trained for 50 epochs using mean squared error loss and Adam optimizer.
Reconstruction error (MSE) calculated for each data point.
Points with errors above the 95th percentile are flagged as anomalies.

Results & Analysis:
Both models detected overlapping but distinct sets of anomalies, highlighting different aspects of irregular behavior.
Isolation Forest focused on transactional volume and frequency irregularities.
Autoencoder identified unusual relationships between owners, buyers, and sales values.
Visualization plots (scatter plots and histograms) illustrate the anomaly distribution and feature relationships for interpretability.

How to Use:
Load the nft_sales.csv dataset.
Run the preprocessing steps to clean and engineer features.
Fit both Isolation Forest and Autoencoder models on normalized features.
Analyze flagged anomalies and visualize using the provided plotting scripts.
Adjust contamination and threshold levels as needed based on domain knowledge.

Dependencies:
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
keras (TensorFlow backend)

This approach offers a robust framework to detect unusual patterns in NFT market data by combining statistical and neural network methods, helping analysts flag suspicious transactions and better understand NFT trading behavior.
