import React from 'react';

const drugImages = {
  'aspirin': 'A',
  'ibuprofen': 'I',
  'paracetamol': 'P',
  'warfarin': 'W',
  'simvastatin': 'S'
};

const colors = ['#ff7675', '#74b9ff', '#55efc4', '#ffeaa7', '#a29bfe', '#fd79a8'];

const DrugCard = ({ title, options, selectedValue, onChange }) => {
  const getRandomColor = () => {
    return colors[Math.floor(Math.random() * colors.length)];
  };

  const imageStyle = {
    backgroundColor: selectedValue ? getRandomColor() : '#f0f0f0'
  };

  return (
    <div className="drug-card">
      <h2>{title}</h2>
      <div className="drug-image" style={imageStyle}>
        {selectedValue ? drugImages[selectedValue] : 'Drug Image'}
      </div>
      <select 
        value={selectedValue} 
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">Select a drug</option>
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
};

export default DrugCard;