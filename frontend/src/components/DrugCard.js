import React, { useState, useEffect, useRef } from 'react';
import Select from 'react-select';
import Modal from 'react-modal';
import OCL from 'openchemlib';  

const DrugCard = ({ title, options, selectedValue, onChange, otherSelectedValue }) => {
  const [sdfData, setSdfData] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const canvasRef = useRef(null);
  const modalCanvasRef = useRef(null);

  useEffect(() => {
    if (selectedValue) {
      fetch(`http://localhost:8000/getSdf/${selectedValue}`)
        .then(res => res.text())
        .then(data => setSdfData(data))
        .catch(err => {
          console.error('Error fetching SDF:', err);
          setSdfData('');
        });
    } else {
      setSdfData('');
    }
  }, [selectedValue]);

  useEffect(() => {
    if (sdfData && canvasRef.current) {
      try {
        const molecule = OCL.Molecule.fromMolfile(sdfData);
        const svg = molecule.toSVG(300, 300);
        canvasRef.current.innerHTML = svg;
      } catch (err) {
        console.error('Error rendering molecule:', err);
        canvasRef.current.innerHTML = '<p>Unable to render molecule</p>';
      }
    } else if (canvasRef.current) {
      canvasRef.current.innerHTML = '';
    }
  }, [sdfData]);

  useEffect(() => {
  if (isModalOpen && sdfData) {
    const timeout = setTimeout(() => {
      if (modalCanvasRef.current) {
        try {
          const molecule = OCL.Molecule.fromMolfile(sdfData);
          const svg = molecule.toSVG(800, 800);
          modalCanvasRef.current.innerHTML = svg;
        } catch (err) {
          console.error('Error rendering molecule in modal:', err);
          modalCanvasRef.current.innerHTML = '<p>Unable to render molecule</p>';
        }
      }
    }, 100); // slight delay to let modal DOM appear
    return () => clearTimeout(timeout);
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
          cursor: 'pointer'
        }}
        onClick={() => setIsModalOpen(true)}
      >
        {!sdfData && selectedValue && <p>Loading molecule...</p>}
      </div>

      <Modal
        isOpen={isModalOpen}
        onRequestClose={() => setIsModalOpen(false)}
        contentLabel="Molecule Viewer"
        style={{
          overlay: {
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            zIndex: 1000,
            overflow: 'hidden'
          },
          content: { 
            position: 'absolute',
            top: '50%',
            left: '50%',
            right: 'auto',
            bottom: 'auto',
            transform: 'translate(-50%, -50%)',
            padding: '0',
            border: 'none',
            background: '#f9f9f9',
            borderRadius: '10px',
            overflow: 'hidden',
            width: '650px',
            height: '700px',
            display: 'flex',
            flexDirection: 'column'
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
          <h2 style={{ margin: 0 }}>Molecule Viewer</h2>
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
            flex: 1,
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden'
          }}
        >
          {!sdfData && <p>Loading molecule...</p>}
        </div>
      </Modal>
    </div>
  );
};

export default DrugCard;
