"use client";

import { useGraph } from "@/app/GraphContext"; // Chemin vers GraphContext
import { useState, useEffect } from "react";

// Définir les types de clés pour les graphiques
type GraphType = 'redditSentimentMap' | 'redditCircularGraph' | 'twitterSentimentMap' | 'twitterCircularGraph' | 'redditSentimentEvolution' | 'twitterSentimentEvolution' | 'twitterWordcloud' | 'redditWordcloud' | 'redditWordcount' | 'postEvolution';

const ResultsPage = () => {
  const { graphs, numberPosts, selectedYear, selectedCountry } = useGraph(); // Utilisation du contexte
  const [graphTypes, setGraphTypes] = useState<GraphType[]>([]); // Définir un état pour les types de graphiques
  const [graphData, setGraphData] = useState<{ [key in GraphType]?: string }>({}); // Définir un état pour les données des graphiques
  const [currentGraphIndex, setCurrentGraphIndex] = useState<number>(0); // État pour l'index du graphique actuel

  const graphDescriptions: Record<GraphType, string> = {
    'redditSentimentMap': 'This sentiment map shows the sentiment scores across the world on various topics.',
    'redditCircularGraph': 'This circular graph illustrates the distribution of positive, neutral, and negative sentiments regarding the general sentiment.',
    'twitterSentimentMap': 'This sentiment map shows the sentiment scores on Twitter across different regions.',
    'twitterCircularGraph': 'This circular graph on Twitter illustrates the distribution of positive, neutral, and negative sentiments regarding the general sentiment.',
    'redditSentimentEvolution': 'This graph shows the evolution of sentiment scores over time on Reddit.',
    'twitterSentimentEvolution': 'This graph shows the evolution of sentiment scores over time on Twitter.',
    'twitterWordcloud': 'This word cloud shows the most frequent words used in tweets.',
    'redditWordcloud': 'This word cloud shows the most frequent words used in Reddit posts.',
    'redditWordcount': 'This graph shows the word count distribution in Reddit posts.',
    'postEvolution': 'This graph shows the evolution of the number of posts over time.',
  };

  useEffect(() => {
    if (graphs) {
      // Extraire les types de graphiques disponibles
      const availableGraphTypes: GraphType[] = Object.keys(graphs) as GraphType[];
  
      // Définir les types de graphiques disponibles et les données
      setGraphTypes(availableGraphTypes);
   
      setGraphData(
        availableGraphTypes.reduce((acc, type) => {
          acc[type] = `data:image/png;base64,${graphs[type]}`;
          return acc;
        }, {} as { [key in GraphType]?: string })
      );
    }
  }, [graphs]);

  const handleNext = () => {
    setCurrentGraphIndex((prevIndex) => (prevIndex + 1) % graphTypes.length);
  };

  const handlePrev = () => {
    setCurrentGraphIndex((prevIndex) => (prevIndex - 1 + graphTypes.length) % graphTypes.length);
  };

  // Construct the key for numberPosts based on the current graph type, year, and country
  const currentGraphKey = `${graphTypes[currentGraphIndex]}`;

  return (
    <div className="bg-gray-100 min-h-screen p-10">
      <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-lg border border-accent relative">
        <h1 className="text-3xl font-bold text-center text-accent mb-6">Results for {selectedYear}</h1>

        {graphTypes.length > 0 ? (
          <div className="relative">
            <button
              onClick={handlePrev}
              className="absolute left-0 top-1/2 transform -translate-y-1/2 bg-gray-300 p-2 rounded-full shadow-lg z-10"
            >
              &larr;
            </button>
            <div className="flex justify-center items-center min-h-64">
              <div key={graphTypes[currentGraphIndex]} className="min-w-full">
                <h2 className="text-2xl font-semibold text-center mb-4">
                  {graphTypes[currentGraphIndex].replace('_', ' ').toUpperCase()}
                </h2>
                <p className="text-lg text-gray-700 mb-4">
                  {graphDescriptions[graphTypes[currentGraphIndex].split('_')[0] as GraphType]} 
                </p>
                <div className="flex justify-center">
                  {graphData[graphTypes[currentGraphIndex]] ? (
                    <>
                      <img
                        src={graphData[graphTypes[currentGraphIndex]]}
                        alt={graphTypes[currentGraphIndex]}
                        className="max-w-full h-auto rounded-lg"
                      />
                      <a
                        href={graphData[graphTypes[currentGraphIndex]]}
                        download={`${graphTypes[currentGraphIndex]}_${selectedYear}.png`}
                        className="btn btn-primary btn-accent text-white mt-4 absolute top-[-25px] right-[40px]" // Positionnez le bouton ici
                      >
                        Download Graph
                      </a>
                    </>
                  ) : (
                    <p className="text-center text-gray-500">Loading...</p>
                  )}
                </div>

                {(graphTypes[currentGraphIndex] === 'redditCircularGraph' || graphTypes[currentGraphIndex] === 'twitterCircularGraph') && selectedCountry && (
                  <div className="text-center mt-4">
                    <p className="text-md text-gray-600">Country: {selectedCountry}</p>
                  </div>
                )}
              </div>
            </div>
            <button
              onClick={handleNext}
              className="absolute right-0 top-1/2 transform -translate-y-1/2 bg-gray-300 p-2 rounded-full shadow-lg z-10"
            >
              &rarr;
            </button>

            <div className="absolute bottom-[-70px] right-[-100px] stats shadow">
              <div className="stat">
                <div className="stat-title">Total Posts</div>
                <div className="stat-value text-center text-accent">{numberPosts[currentGraphKey]}</div>
              </div>
            </div>

          </div>
        ) : (
          <p className="text-center text-gray-500">No graphs available.</p>
        )}
      </div>
    </div>
  );
};

export default ResultsPage;
