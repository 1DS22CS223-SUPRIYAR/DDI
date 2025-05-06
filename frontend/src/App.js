import React, { useState, useEffect } from 'react';
import './styles/App.css';
import DrugCard from './components/DrugCard';
import Results from './components/Results';

function App() {
  const [selectedDrugs, setSelectedDrugs] = useState({ drug1: '', drug2: '' });
  const [results, setResults] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [drugOptions, setDrugOptions] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch drug list from backend (FastAPI)
  useEffect(() => {
    const fetchDrugs = async () => {
      setLoading(true);
      try {
        const res = await fetch('http://localhost:8000/drugs', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }

        const data = await res.json();
        console.log(data);

        // Convert object to array of values
        const drugsArray = Object.values(data.drugs);
        setDrugOptions(drugsArray);
        setError(null);
      } catch (err) {
        console.error('Error fetching drug data:', err);
        setError('Failed to load drug list. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchDrugs();
  }, []);

  const interactions = {
    aspirin_ibuprofen: {
      type: 'Moderate Interaction',
      desc: 'Increased risk of gastrointestinal bleeding when these two NSAIDs are taken together.',
      accuracy: 92,
    },
    aspirin_warfarin: {
      type: 'Major Interaction',
      desc: 'Aspirin may increase the anticoagulant effect of warfarin, increasing bleeding risk.',
      accuracy: 97,
    },
    simvastatin_warfarin: {
      type: 'Minor Interaction',
      desc: 'Simvastatin may slightly increase the effect of warfarin. Monitor INR.',
      accuracy: 85,
    },
    default: {
      type: 'No Interaction',
      desc: 'These drugs can be safely taken together.',
      accuracy: 95,
    },
  };

  const handleDrugChange = (drug, value) => {
    setSelectedDrugs((prev) => ({
      ...prev,
      [drug]: value,
    }));
    setShowResults(false);
  };

  const handlePredict = () => {
    if (!selectedDrugs.drug1 || !selectedDrugs.drug2) {
      alert('Please select both drugs');
      return;
    }

    const interactionKey1 = `${selectedDrugs.drug1.toLowerCase()}_${selectedDrugs.drug2.toLowerCase()}`;
    const interactionKey2 = `${selectedDrugs.drug2.toLowerCase()}_${selectedDrugs.drug1.toLowerCase()}`;

    const interaction =
      interactions[interactionKey1] ||
      interactions[interactionKey2] ||
      interactions['default'];

    setResults(interaction);
    setShowResults(true);
  };

  return (
    <div className="container">
      <header>
        <h1>Drug Interaction Predictor</h1>
        <p className="subtitle">Check potential interactions between medications</p>
      </header>

      {loading && <p>Loading drugs...</p>}
      {error && <p className="error-message">{error}</p>}

      <div className="drug-selector">
        <DrugCard
          title="Drug 1"
          options={drugOptions}
          selectedValue={selectedDrugs.drug1}
          onChange={(value) => handleDrugChange('drug1', value)}
        />

        <DrugCard
          title="Drug 2"
          options={drugOptions}
          selectedValue={selectedDrugs.drug2}
          onChange={(value) => handleDrugChange('drug2', value)}
        />
      </div>

      <button
        className="predict-btn"
        onClick={handlePredict}
        disabled={!selectedDrugs.drug1 || !selectedDrugs.drug2}
      >
        Predict Interaction
      </button>

      {showResults && results && (
        <Results
          interactionType={results.type}
          description={results.desc}
          accuracy={results.accuracy}
        />
      )}

      <div className="footer">
        <p>© 2023 Drug Interaction Predictor. For educational purposes only.</p>
      </div>
    </div>
  );
}

export default App;
