import React, { useState, useEffect, useRef } from 'react';
import Select from 'react-select';
import Modal from 'react-modal';
import OCL from 'openchemlib';

const DrugCard = ({ title, options, selectedValue, onChange, otherSelectedValue }) => {
  const [sdfData, setSdfData] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const canvasRef = useRef(null);
  const modalCanvasRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (selectedValue) {
      setIsLoading(true);
      fetch(`http://localhost:8000/getSdf/${selectedValue}`)
        .then(res => res.text())
        .then(data => {
          setSdfData(data);
          setIsLoading(false);
        })
        .catch(err => {
          console.error('Error fetching SDF:', err);
          setSdfData('');
          setIsLoading(false);
        });
    } else {
      setSdfData('');
    }
  }, [selectedValue]);

  const renderMolecule = (canvasElement, sdf, width, height) => {
    if (!canvasElement || !sdf) return;
    
    try {
      const molecule = OCL.Molecule.fromMolfile(sdf);
      // Clear previous content
      canvasElement.innerHTML = '';
      
      // Create SVG with proper viewBox and preserveAspectRatio
      const svg = molecule.toSVG(width, height, undefined, {
        viewBox: `0 0 ${width} ${height}`,
        preserveAspectRatio: 'xMidYMid meet'
      });
      
      // Center the SVG in its container
      const container = document.createElement('div');
      container.style.display = 'flex';
      container.style.justifyContent = 'center';
      container.style.alignItems = 'center';
      container.style.width = '100%';
      container.style.height = '100%';
      container.innerHTML = svg;
      
      canvasElement.appendChild(container);
    } catch (err) {
      console.error('Error rendering molecule:', err);
      canvasElement.innerHTML = '<p>Unable to render molecule</p>';
    }
  };

  useEffect(() => {
    renderMolecule(canvasRef.current, sdfData, 300, 300);
  }, [sdfData]);

  useEffect(() => {
    if (isModalOpen) {
      // Small delay to ensure modal is fully rendered
      const timer = setTimeout(() => {
        renderMolecule(modalCanvasRef.current, sdfData, 600, 600);
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [isModalOpen, sdfData]);

  const selectOptions = options.map(drug => ({
    value: drug.id,
    label: drug.name,
    isDisabled: drug.id === otherSelectedValue
  }));

  return (
    <div className="drug-card">
      <h2>{title}</h2>

      <Select
        options={selectOptions}
        value={selectOptions.find(opt => opt.value === selectedValue) || null}
        onChange={(selected) => onChange(selected ? selected.value : '')}
        placeholder="Select a drug"
        isSearchable
        menuPlacement="bottom"
      />

      <div
        ref={canvasRef}
        style={{
          marginTop: '10px',
          width: '300px',
          height: '300px',
          border: '1px solid #ccc',
          background: '#fff',
          cursor: 'pointer',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center'
        }}
        onClick={() => setIsModalOpen(true)}
      >
        {isLoading ? (
          <p>Loading molecule...</p>
        ) : !sdfData && selectedValue ? (
          <p>No structure available</p>
        ) : !selectedValue ? (
          <p>Select a drug to view structure</p>
        ) : null}
      </div>

      <Modal
        isOpen={isModalOpen}
        onRequestClose={() => setIsModalOpen(false)}
        contentLabel="Molecule Viewer"
        style={{
          overlay: {
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            zIndex: 1000
          },
          content: { 
            top: '50%',
            left: '50%',
            right: 'auto',
            bottom: 'auto',
            transform: 'translate(-50%, -50%)',
            padding: '0',
            border: 'none',
            background: '#f9f9f9',
            borderRadius: '10px',
            width: '650px',
            height: '700px',
            maxWidth: '90vw',
            maxHeight: '90vh'
          }
        }}
      >
        <div style={{
          padding: '20px',
          borderBottom: '1px solid #eee',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h2 style={{ margin: 0 }}>Molecule Structure</h2>
          <button 
            onClick={() => setIsModalOpen(false)}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              cursor: 'pointer',
              padding: '0 10px'
            }}
          >
            ×
          </button>
        </div>
        <div
          ref={modalCanvasRef}
          style={{
            width: '100%',
            height: 'calc(100% - 61px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'auto',
            padding: '20px'
          }}
        >
          {isLoading ? (
            <p>Loading molecule...</p>
          ) : !sdfData ? (
            <p>No structure available</p>
          ) : null}
        </div>
      </Modal>
    </div>
  );
};

export default DrugCard;