import argparse
import sys
import logging
import numpy as np
from scipy.stats import norm
from src.analysis.market_edge import analyze_market_question
from src.collectors.sea_ice import predict_sea_ice_minimum
from src.data_loader import get_merged_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_temperature_analysis(year, threshold, market_prob):
    print(f"\n--- Temperature Anomaly Analysis for {year} > {threshold}C ---")
    prob, edge = analyze_market_question(year, threshold, market_prob)
    if prob is not None:
        print(f"Model Probability: {prob:.1%}")
        if market_prob is not None:
            # Re-print edge from returned value to be safe, but analyze_market_question already prints it.
            pass
    else:
        print("Could not generate prediction.")

def run_sea_ice_analysis(year, threshold=None):
    print(f"\n--- Arctic Sea Ice Minimum Analysis for {year} ---")
    pred, std = predict_sea_ice_minimum(year)
    print(f"Predicted Minimum Extent: {pred:.2f} +/- {std:.2f} M km2")

    if threshold is not None:
        # Probability of being lower than threshold (Record Low?)
        z_score = (threshold - pred) / std
        # Prob(X < threshold)
        prob_lower = norm.cdf(z_score)
        print(f"Probability of being < {threshold} M km2: {prob_lower:.1%}")

def main():
    parser = argparse.ArgumentParser(description="Climate Prediction Market System")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Temperature Command
    temp_parser = subparsers.add_parser('temp', help='Analyze Global Temperature Anomaly')
    temp_parser.add_argument('--year', type=int, required=True, help='Target Year')
    temp_parser.add_argument('--threshold', type=float, required=True, help='Threshold in Celsius')
    temp_parser.add_argument('--market-prob', type=float, help='Current Market Probability (0-1)')

    # Sea Ice Command
    ice_parser = subparsers.add_parser('ice', help='Analyze Arctic Sea Ice Minimum')
    ice_parser.add_argument('--year', type=int, required=True, help='Target Year')
    ice_parser.add_argument('--threshold', type=float, help='Record Low Threshold (M km2)')

    # Update Data Command
    data_parser = subparsers.add_parser('update', help='Update Data')

    args = parser.parse_args()

    if args.command == 'update':
        logging.info("Updating data...")
        try:
            get_merged_data(download=True)
            logging.info("Data updated successfully.")
        except Exception as e:
            logging.error(f"Failed to update data: {e}")

    elif args.command == 'temp':
        run_temperature_analysis(args.year, args.threshold, args.market_prob)

    elif args.command == 'ice':
        run_sea_ice_analysis(args.year, args.threshold)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
