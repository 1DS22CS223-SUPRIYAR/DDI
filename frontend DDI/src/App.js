import React, { useState } from 'react';
import './styles/App.css';
import DrugCard from './components/DrugCard';
import Results from './components/Results';

function App() {
  const [selectedDrugs, setSelectedDrugs] = useState({ drug1: '', drug2: '' });
  const [results, setResults] = useState(null);
  const [showResults, setShowResults] = useState(false);

  const drugOptions = [
    { value: 'aspirin', label: 'Aspirin' },
    { value: 'ibuprofen', label: 'Ibuprofen' },
    { value: 'paracetamol', label: 'Paracetamol' },
    { value: 'warfarin', label: 'Warfarin' },
    { value: 'simvastatin', label: 'Simvastatin' },
  ];

  const interactions = {
    'aspirin_ibuprofen': {
      type: 'Moderate Interaction',
      desc: 'Increased risk of gastrointestinal bleeding when these two NSAIDs are taken together.',
      accuracy: 92
    },
    'aspirin_warfarin': {
      type: 'Major Interaction',
      desc: 'Aspirin may increase the anticoagulant effect of warfarin, increasing bleeding risk.',
      accuracy: 97
    },
    'simvastatin_warfarin': {
      type: 'Minor Interaction',
      desc: 'Simvastatin may slightly increase the effect of warfarin. Monitor INR.',
      accuracy: 85
    },
    'default': {
      type: 'No Interaction',
      desc: 'These drugs can be safely taken together.',
      accuracy: 95
    }
  };

  const handleDrugChange = (drug, value) => {
    setSelectedDrugs(prev => ({
      ...prev,
      [drug]: value
    }));
    setShowResults(false);
  };

  const handlePredict = () => {
    if (!selectedDrugs.drug1 || !selectedDrugs.drug2) {
      alert('Please select both drugs');
      return;
    }

    const interactionKey1 = `${selectedDrugs.drug1}_${selectedDrugs.drug2}`;
    const interactionKey2 = `${selectedDrugs.drug2}_${selectedDrugs.drug1}`;
    
    const interaction = interactions[interactionKey1] || 
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
      
      <button className="predict-btn" onClick={handlePredict}>
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