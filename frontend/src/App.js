import React, { useState, useEffect } from 'react';
import './styles/App.css';
import DrugCard from './components/DrugCard';
import Results from './components/Results';
import About from './components/About';

function App() {
  const [selectedDrugs, setSelectedDrugs] = useState({ drug1: '', drug2: '' });
  const [results, setResults] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [drugOptions, setDrugOptions] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState('home');

  // Fetch drug list from backend (FastAPI)
  useEffect(() => {
    const fetchDrugs = async () => {
      console.log("Fetching drugs...");
      setLoading(true);
      try {
        const res = await fetch('http://127.0.0.1:8000/drugs', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        const data = await res.json();
        console.log(data)
        const drugsArray = Object.entries(data.drugs).map(([id, name]) => ({
          id,
          name
        }));
        setDrugOptions(drugsArray);
        setError(null);
      } catch (err) {
        console.error('Error fetching drug data:', err);
        setError('Failed to load drug list. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    if (currentPage === 'home') {
      fetchDrugs();
    }
  }, [currentPage]);

  const handleDrugChange = (drug, value) => {
    setSelectedDrugs((prev) => ({
      ...prev,
      [drug]: value,
    }));
    setShowResults(false);
  };

  const handlePredict = async () => {
    if (!selectedDrugs.drug1 || !selectedDrugs.drug2) {
      alert('Please select both drugs');
      return;
    }
  
    try {
      const response = await fetch(`http://127.0.0.1:8000/predict_interaction?drug1=${selectedDrugs.drug1}&drug2=${selectedDrugs.drug2}`);
      if (!response.ok) {
        throw new Error('Failed to fetch interaction data');
      }
  
      const data = await response.json();
      setResults(data);
      setShowResults(true);

      setTimeout(()=>{
        window.scrollTo({
          top: document.documentElement.scrollHeight,
          behavior: 'smooth',
        })
      }, 100)

    } catch (error) {
      console.error('Error fetching interaction:', error);
      alert('An error occurred while fetching interaction data');
    }
  };

  return (
    <div className="app-container">
      <nav className="top-nav">
        <div className="app-name">Drugram</div>
        <div className="nav-links">
          <button 
            className={`nav-link ${currentPage === 'home' ? 'active' : ''}`}
            onClick={() => {
              setCurrentPage('home')
              window.location.reload();
            }}
          >
            Home
          </button>
          <button 
            className={`nav-link ${currentPage === 'about' ? 'active' : ''}`}
            onClick={() => setCurrentPage('about')}
          >
            About
          </button>
        </div>
      </nav>

      <div className="main-content">
        {currentPage === 'home' ? (
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
                otherSelectedValue={selectedDrugs.drug2}
              />

              <DrugCard
                title="Drug 2"
                options={drugOptions}
                selectedValue={selectedDrugs.drug2}
                onChange={(value) => handleDrugChange('drug2', value)}
                otherSelectedValue={selectedDrugs.drug1}
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
          </div>
        ) : (
          <About />
        )}
      </div>

      <div className="footer">
        <p>© 2025 Drug Interaction Predictor. For educational purposes only.</p>
      </div>
    </div>
  );
}

export default App;