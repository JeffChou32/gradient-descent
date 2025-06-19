import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


class WineQualitySGDModel:
    def __init__(self, data_url, separator=';', test_size=0.2, random_state=42):
        self.data_url = data_url
        self.separator = separator
        self.test_size = test_size
        self.random_state = random_state

        self.df = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.scaler = StandardScaler()
        self.model = None
        self.log_file = "sgd_log.txt"

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.data_url, sep=self.separator).dropna().drop_duplicates()
        X = self.df.drop('quality', axis=1)
        y = self.df['quality']
        
        X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )

    def plot_data(self):        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr().round(2), annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        sns.histplot(self.df['quality'], bins=10, kde=True)
        plt.title('Wine Quality Distribution')
        plt.savefig('quality_distribution.png')
        plt.close()
        
        top_features = self.df.corr()['quality'].abs().sort_values(ascending=False).index[1:3]
        for feature in top_features:
            sns.scatterplot(data=self.df, x=feature, y='quality', alpha=0.5)
            plt.title(f'{feature} vs Quality')
            plt.savefig(f'scatter_{feature}_vs_quality.png')
            plt.close()

    def train_model(self, alpha, max_iter):
        self.model = SGDRegressor(alpha=alpha, max_iter=max_iter, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        explained_var = explained_variance_score(self.y_test, y_test_pred)

        return {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "explained_variance": explained_var,
            "coefficients": self.model.coef_,
            "intercept": self.model.intercept_
        }

    def run_experiments(self, alphas, iterations):
        best_score = float('-inf')
        best_result = {}

        with open(self.log_file, "w") as log:
            for alpha in alphas:
                for max_iter in iterations:
                    self.train_model(alpha, max_iter)
                    metrics = self.evaluate_model()

                    log.write(
                        f"alpha={alpha}, max_iter={max_iter}, "
                        f"train_rmse={metrics['train_rmse']:.4f}, "
                        f"test_rmse={metrics['test_rmse']:.4f}, "
                        f"test_r2={metrics['test_r2']:.4f}, "
                        f"explained_variance={metrics['explained_variance']:.4f}\n"
                    )

                    if metrics['test_r2'] > best_score:
                        best_score = metrics['test_r2']
                        best_result = {
                            "alpha": alpha,
                            "max_iter": max_iter,
                            **metrics
                        }

        return best_result

    def display_results(self, results):
        print("\nBest Model Results")
        print("------------------")
        for key, value in results.items():
            if isinstance(value, (float, np.float64, np.float32)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/JeffChou32/wine-dataset/refs/heads/main/winequality-white.csv'

    model = WineQualitySGDModel(data_url=url)
    model.load_and_preprocess_data()
    model.plot_data()
    best_results = model.run_experiments(alphas=[0.0001, 0.001, 0.01], iterations=[500, 1000, 2000])
    model.display_results(best_results)
