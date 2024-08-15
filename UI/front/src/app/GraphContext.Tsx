"use client"

import { createContext, useContext, useState, ReactNode } from 'react';
import React from 'react';

// Define the type for graphs
interface Graphs {
  [key: string]: string | null;
}

interface GraphNumber {
  [key: string]: number | null;
}

interface GraphContextType {
  graphs: Graphs;
  setGraphs: (graphs: Graphs) => void;
  numberPosts: GraphNumber;
  setNumber: (number: GraphNumber) => void;
  selectedYear: string | null;
  setSelectedYear: (year: string | null) => void;
  selectedCountry: string | null;
  setSelectedCountry: (country: string | null) => void;
}

const GraphContext = createContext<GraphContextType | undefined>(undefined);

export const GraphProvider = ({ children }: { children: ReactNode }) => {
  const [graphs, setGraphs] = useState<Graphs>({});
  const [selectedYear, setSelectedYear] = useState<string | null>(null);
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [numberPosts, setNumber] = useState<GraphNumber>({});
  return (
    <GraphContext.Provider value={{
      graphs,
      setGraphs,
      numberPosts,
      setNumber,
      selectedYear,
      selectedCountry,
      setSelectedYear,
      setSelectedCountry,
    }}>
      {children}
    </GraphContext.Provider>
  );
};

export const useGraph = () => {
  const context = useContext(GraphContext);
  if (context === undefined) {
    throw new Error('useGraph must be used within a GraphProvider');
  }
  return context;
};
