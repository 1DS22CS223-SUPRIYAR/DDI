import React from 'react';

const drugImages = {
  'aspirin': 'A',
  'ibuprofen': 'I',
  'paracetamol': 'P',
  'warfarin': 'W',
  'simvastatin': 'S'
};

const DrugCard = ({ title, options, selectedValue, onChange, otherSelectedValue }) => {
  return (
    <div className="drug-card">
      <h2>{title}</h2>
      <div className="drug-image">
        {selectedValue ? drugImages[selectedValue] || selectedValue.charAt(0).toUpperCase() : 'Drug Image'}
      </div>
      <select 
        value={selectedValue} 
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">Select a drug</option>
        {options.map((drug, index) => (
          <option 
            key={`${drug}-${index}`} 
            value={drug} 
            disabled={drug === otherSelectedValue}
          >
            {drug}
          </option>
        ))}
      </select>
    </div>
  );
};

export default DrugCard;
