from search_engine.experiment import ScaledExperiment
from search_engine.config import get_config

def main():
    config = get_config()
    experiment = ScaledExperiment(config)
    results = experiment.run()
    
    print("\nExperiment Results:")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Time Taken: {results['time_taken']:.2f} seconds")
    print(f"Number of Features Found: {len(results['augplan'])}")

if __name__ == "__main__":
    main()