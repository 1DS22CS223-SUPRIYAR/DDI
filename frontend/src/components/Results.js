import React, { useEffect, useRef } from 'react';

const Results = ({ interactionType, description, accuracy }) => {
  const accuracyFillRef = useRef(null);

  useEffect(() => {
    if (accuracyFillRef.current) {
      accuracyFillRef.current.style.width = `${accuracy*100}%`;
      
      // Set color based on accuracy
      if (accuracy*100 > 90) {
        accuracyFillRef.current.style.backgroundColor = 'var(--success)';
      } else if (accuracy*100 > 75) {
        accuracyFillRef.current.style.backgroundColor = 'var(--warning)';
      } else {
        accuracyFillRef.current.style.backgroundColor = 'var(--danger)';
      }
    }
  }, [accuracy]);

  return (
    <div className="results">
      <h2>Interaction Results</h2>
      
      <div className="result-item">
        <h3>Interaction Type</h3>
        <p>{interactionType}</p>
      </div>
      
      <div className="result-item">
        <h3>Description</h3>
        <p>{description}</p>
      </div>
      
      <div className="result-item">
        <h3>Prediction Accuracy</h3>
        <p>{accuracy}%</p>
        <div className="accuracy-meter">
          <div className="accuracy-fill" ref={accuracyFillRef}></div>
        </div>
      </div>
    </div>
  );
};

export default Results;