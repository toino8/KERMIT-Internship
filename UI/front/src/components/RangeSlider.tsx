import React, { useState } from 'react';
import { Range } from 'react-range';

interface RangeSliderProps {
  onRangeChange: (newRange: number[]) => void;
}

function RangeSlider({ onRangeChange }: RangeSliderProps) {
  const [values, setValues] = useState<number[]>([2008, 2024]);

  const handleChange = (newValues: number[]) => {
    console.log('New values:', newValues);
    setValues(newValues);
    onRangeChange(newValues);
  };

  return (
    <div className="flex justify-center items-center flex-col">
      <div className="w-80 m-10">
        <Range
          step={1}
          min={2008}
          max={2024}
          values={values}
          onChange={handleChange}
          renderTrack={({ props, children }) => (
            <div
              {...props}
              className="h-4 w-full bg-gray-200 rounded-full"
            >
              {children}
            </div>
          )}
          renderThumb={({ props }) => (
            <div
              {...props}
              className="h-7 w-7 bg-accent rounded-full shadow-md cursor-pointer"
            />
          )}
        />
      </div>
      <div className="text-center">
        <span className="text-sm font-semibold">Selected Range: {values[0]} - {values[1]}</span>
      </div>
    </div>
  );
}

export default RangeSlider;
