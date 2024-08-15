"use client";

import { useState, useEffect } from "react";
import axios from 'axios';
import { useRouter } from 'next/navigation';
import { ChooseBar } from "@/components/chooseBar";
import RangeSlider from "@/components/RangeSlider";
import { useGraph } from "@/app/GraphContext";
import { useNavContext } from "../NavContext";

// Define the interface for the country
interface Country {
  name: string;
}

interface Language {
  [name: string]: string;
}

export default function HomePage() {
  const router = useRouter();
  const { setCurrentPage } = useNavContext();
  const { setGraphs, setNumber, setSelectedCountry } = useGraph(); // Use the context

  // Global state
  const [loading, setLoading] = useState<boolean>(false); 
  const [selectedYearMonths, setSelectedYearMonthsPage] = useState<{ year: string; month: string }[]>([]);

  const [rangeYearEvolution, setRangeYearEvolution] = useState<number[]>([]);
  const [theme, setTheme] = useState<string>('Climate Change');

  
  // State for Twitter
  const [twitterSentimentMap, setTwitterSentimentMap] = useState<boolean>(false);
  const [twitterCircularGraph, setTwitterCircularGraph] = useState<boolean>(false);
  const [twitterSearchTermGraph, setTwitterSearchTermGraph] = useState<string>('');
  const [twitterFilteredLanguagesGraph, setTwitterFilteredLanguagesGraph] = useState<Language[]>([]);
  const [selectedTwitterLanguageGraph, setSelectedTwitterLanguageGraph] = useState<string | null>(null);
  const [twitterEvolutionChecked, setTwitterEvolutionChecked] = useState<boolean>(false);
  const [twitterWordCloud, setTwitterWordCloud] = useState<boolean>(false);
  const [twitterFilteredLanguagesWordCloud, setTwitterFilteredLanguagesWordCloud] = useState<Language[]>([]);
  const [searchTwitterWordCloud, setSearchTwitterWordCloud] = useState<string>('');
  const [selectedTwitterLanguageWordCloud, setSelectedTwitterLanguageWordCloud] = useState<string | null>(null);
  const [twitterSearchTermWordCount, setTwitterSearchTermWordCount] = useState<string>('');
  const [twitterWordCount, setTwitterWordCount] = useState<boolean>(false);
  const [filteredTwitterWordCountLanguages, setFilteredTwitterWordCountLanguages] = useState<Language[]>([]);
  const [selectedLanguageTwitterWordCount, setSelectedLanguageTwitterWordCount] = useState<string | null>(null);

  // State for Reddit
  const [selectedCountryWordCloudReddit, setSelectedCountryWordCloudReddit] = useState<string>('');
  const [filteredRedditGraphCountries, setFilteredRedditGraphCountries] = useState<Country[]>([]);
  const [redditSentimentMap, setRedditSentimentMap] = useState<boolean>(false);
  const [redditCircularGraph, setRedditCircularGraph] = useState<boolean>(false);
  const [redditWordCloud, setRedditWorcloud] = useState<boolean>(false);
  const [redditEvolutionChecked, setRedditEvolutionChecked] = useState<boolean>(false);
  const [RedditSearchTermGraph, setRedditSearchTermGraph] = useState<string>('');
  const [selectedCountryRedditCircular, setSelectedCountryRedditCircular] = useState<string>('');
  const [RedditSearchTermWordCloud, setRedditSearchTermWordCloud] = useState<string>('');
  const [filteredRedditWordCloudCountries, setFilteredRedditWordCloudCountries] = useState<Country[]>([]);
  const [selectedCountryWordCountReddit, setSelectedCountryRedditWordCount] = useState<string >('');
  const [searchRedditWordCount, setSearchRedditWordCount] = useState<string>('');
  const [redditWordCount, setRedditWordCount] = useState<boolean>(false);
  const [filteredRedditWordCountCountries, setFilteredRedditWordCountCountries] = useState<Country[]>([]);

  // State for search and filters
  const [postEvolutionSearchTerm, setPostEvolutionSearchTerm] = useState<string>('');
  const [redditPostEvolutionChecked, setRedditPostEvolutionChecked] = useState<boolean>(false);
  const [postEvolution, setPostEvolution] = useState<boolean>(false);
  const [rangeYearPostEvolution, setRangeYearPostEvolution] = useState<number[]>([]);
  const [filteredCountriesPostEvolution, setFilteredCountriesPostEvolution] = useState<Country[]>([]);
  const [selectedPostEvolutionCountry, setSelectedPostEvolutionCountry] = useState<string>('');
  const [evolutionSearchTerm, setEvolutionSearchTerm] = useState<string>('');
  const [sentimentEvolution, setSentimentEvolution] = useState<boolean>(false);
  const [languages, setLanguages] = useState<Language[]>([]);
  const [filteredCountriesEvolution, setFilteredCountriesEvolution] = useState<Country[]>([]);
  const [selectedCountryEvolution, setSelectedCountryEvolution] = useState<string>('');
  const [filteredCountriesAnalyzed, setFiteredCountriesAnalyzed] = useState<Country[]>([]);
  const [filteredCountriesData, setFilteredCountriesData] = useState<Country[]>([]);

 
  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  const handleRangeChangeEvolution = (newRange: number[]) => {
    setRangeYearEvolution(newRange);
  };

  const handleRangeChangePostEvolution = (newRange: number[]) => {
    setRangeYearPostEvolution(newRange);
  }


  const renderSelectedRange = () => {
    if (selectedYearMonths.length === 0) {
      return null;
    }
  
    // Sort the selectedYearMonths array by year and month
    const sortedYearMonths = [...selectedYearMonths].sort((a, b) => {
      const yearDiff = parseInt(a.year) - parseInt(b.year);
      if (yearDiff !== 0) return yearDiff;
      return months.indexOf(a.month) - months.indexOf(b.month);
    });
  
    const start = sortedYearMonths[0];
    const end = sortedYearMonths[sortedYearMonths.length - 1];
  
    return `${start.month} ${start.year} - ${end.month} ${end.year}`;
  };



  useEffect(() => {
    const fetchCountries = async () => {
      const payload = {
        yearMonths: selectedYearMonths,
        country: selectedCountryRedditCircular,
        rangeYear: rangeYearEvolution,
        countryEvolution: selectedCountryEvolution,
        redditSentimentMap: redditSentimentMap,
        redditCircularGraph: redditCircularGraph,
        redditWordCloud: redditWordCloud,
        redditSentimentEvolution: redditEvolutionChecked,
        redditWordCloudCountry: selectedCountryWordCloudReddit,
        redditGraphCountry: selectedCountryRedditCircular,
        redditWordCount: redditWordCount,
        redditWordCountCountry: selectedCountryWordCountReddit,
        twitterWordCount: twitterWordCount,
        twitterWordCountLanguage: selectedLanguageTwitterWordCount,
        twitterSentimentMap: twitterSentimentMap,
        twitterCircularGraph: twitterCircularGraph,
        twitterLanguageGraph: selectedTwitterLanguageGraph,
        twitterSentimentEvolution: twitterEvolutionChecked,
        twitterWordCloud: twitterWordCloud,
        twitterGraphLanguage: selectedTwitterLanguageWordCloud,
        twitterWordCloudLanguage: selectedTwitterLanguageWordCloud,
        postEvolution: postEvolution,
        postEvolutionCountry: selectedPostEvolutionCountry,
        rangeYearPostEvolution: rangeYearPostEvolution,
        theme_chosen: theme.replace(/ /g, '').toLowerCase(),
      };
  
      try {
        const response = await axios.post('http://localhost:5000/run-script', 
          payload,
          { headers: { 'Content-Type': 'application/json' } }
        );
  
        // Get the data
        const countriesData = response.data.countries_data;
        const countriesAnalyzed = response.data.countries_analyzed;
  
        // Fetch countries from the local JSON file
        const countriesResponse = await fetch('/countries.json'); // Relative path to the JSON file in the public directory
        if (!countriesResponse.ok) {
          throw new Error('Network response was not ok');
        }
        const countriesJson = await countriesResponse.json(); // Assuming the JSON is structured as an array of objects
  
        // Filter the JSON data based on countries_data and countries_analyzed
        const filteredCountriesData = countriesJson.filter((country: Country) => {
          const countryKey = country.name.toLowerCase().replace(/\s+/g, ''); // Normalize country names
          return countriesData.some((c: string) => c.toLowerCase().replace(/\s+/g, '') === countryKey) 
                 
        });
        
        const filteredCountriesAnalyzed = countriesJson.filter((country: Country) => {
          const countryKey = country.name.toLowerCase().replace(/\s+/g, ''); // Normalize country names
          return countriesAnalyzed.some((c: string) => c.toLowerCase().replace(/\s+/g, '') === countryKey);
        });

        // Set the filtered countries data
        setFilteredCountriesData(filteredCountriesData);
        setFiteredCountriesAnalyzed(filteredCountriesAnalyzed);
        
      } catch (error) {
        console.error('Error fetching countries from API or JSON:', error);
      }
    };
  
    fetchCountries();
  }, [theme]); // Ensure to include theme in dependencies
  
  
  
  






  const runScript = async () => {
    const payload = {
      yearMonths: selectedYearMonths,
      country: selectedCountryRedditCircular.toLowerCase().replace(/ /g, ''),
      rangeYear: rangeYearEvolution,
      countryEvolution: selectedCountryEvolution.toLowerCase().replace(/ /g, ''),
      redditSentimentMap: redditSentimentMap,
      redditCircularGraph: redditCircularGraph,
      redditWordCloud: redditWordCloud,
      redditSentimentEvolution: redditEvolutionChecked,
      redditWordCloudCountry: selectedCountryWordCloudReddit.toLowerCase().replace(/ /g, ''),
      redditGraphCountry: selectedCountryRedditCircular.toLowerCase().replace(/ /g, ''),
      redditWordCount: redditWordCount,
      redditWordCountCountry: selectedCountryWordCountReddit.toLowerCase().replace(/ /g, ''),
      twitterWordCount: twitterWordCount,
      twitterWordCountLanguage: selectedLanguageTwitterWordCount,
      twitterSentimentMap: twitterSentimentMap,
      twitterCircularGraph: twitterCircularGraph,
      twitterLanguageGraph: selectedTwitterLanguageGraph,
      twitterSentimentEvolution: twitterEvolutionChecked,
      twitterWordCloud: twitterWordCloud,
      twitterGraphLanguage: selectedTwitterLanguageWordCloud,
      twitterWordCloudLanguage: selectedTwitterLanguageWordCloud,
      postEvolution: postEvolution,
      postEvolutionCountry: selectedPostEvolutionCountry.toLowerCase().replace(/ /g, ''),
      rangeYearPostEvolution: rangeYearPostEvolution,
      theme_chosen: theme.replace(/ /g, '').toLowerCase()
    };
    

    try {
      setLoading(true); // Indicate that the process is ongoing
      const response = await axios.post('http://localhost:5000/run-script', payload, {
        headers: { 'Content-Type': 'application/json' }
      });
    
     
      if (response.data.graphs) {
       
        setGraphs(response.data.graphs); // Send all graphs in the context
        setNumber(response.data.number_posts);
        setSelectedCountry(selectedCountryRedditCircular); // Update the context with the selected country
        setCurrentPage('/results'); // Update the current page
        router.push('/results'); // Redirect to the results page
      }
     
    } catch (error) {
      console.error('Error running script:', error);
    } finally {
      setLoading(false); // Reset the loading state
    }
  };

  const handleSentimentChange = (checked: boolean) => {
    setSentimentEvolution(checked);
    setTwitterEvolutionChecked(checked);
    setRedditEvolutionChecked(checked);
  };

  const handlePostEvolutionChange = (checked: boolean) => { 
    setPostEvolution(checked);
    setRedditPostEvolutionChecked(checked);
  };
  const handleRedditEvolutionClick = (checked: boolean) => {
    setRedditEvolutionChecked(checked);
  };

  const handleRedditPostEvolutionClick = (checked: boolean) => {
    setRedditPostEvolutionChecked(checked);
  };

  const handleLanguageChangeGraph = (languageName: string) => {
    setSelectedTwitterLanguageGraph((prev) =>
      prev === languageName ? '' : languageName
    );
   
  };

  const handleLanguageChangeWordCloud = (languageName: string) => {
    setSelectedTwitterLanguageWordCloud((prev) =>
      prev === languageName ? '' : languageName
    );
   
  };




  useEffect(() => {
    const fetchLanguages = async () => {
      try {
        const response = await fetch('/languages.json');
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();

        // Assert the type of data
        const languages = data.languages as { name: string; code: string }[];
        setLanguages(languages);
        setTwitterFilteredLanguagesWordCloud(languages);
        setTwitterFilteredLanguagesGraph(languages);
      
      } catch (error) {
        console.error('Error fetching languages:', error);
      }
    };
    fetchLanguages();
  }, []);

  useEffect(() => {
    setFilteredRedditGraphCountries(
      filteredCountriesAnalyzed.filter(country =>
        country.name.toLowerCase().includes(RedditSearchTermGraph.toLowerCase())
      )
    );
  }, [RedditSearchTermGraph, filteredCountriesAnalyzed]);

  useEffect(() => {
    setFilteredRedditWordCloudCountries(
      filteredCountriesData.filter(country =>
        country.name.toLowerCase().includes(RedditSearchTermWordCloud.toLowerCase())
      )
    );
  }, [RedditSearchTermWordCloud, filteredCountriesData]);

  useEffect(() => {
    setFilteredRedditWordCountCountries(
      filteredCountriesAnalyzed.filter(country =>
        country.name.toLowerCase().includes(searchRedditWordCount.toLowerCase())
      )
    );
   
  }, [searchRedditWordCount, filteredCountriesAnalyzed]);

  useEffect(() => {
    setFilteredCountriesPostEvolution(
      filteredCountriesData.filter(country =>
        country.name.toLowerCase().includes(postEvolutionSearchTerm.toLowerCase())
      )
    );
  }, [postEvolutionSearchTerm, filteredCountriesData]);

  useEffect(() => {
    setFilteredCountriesEvolution(
      filteredCountriesAnalyzed.filter(country =>
        country.name.toLowerCase().includes(evolutionSearchTerm.toLowerCase())
      )
    );

  }, [evolutionSearchTerm, filteredCountriesAnalyzed]);

  // Filtrer les pays en fonction du terme de recherche pour Twitter
  useEffect(() => {
    setTwitterFilteredLanguagesGraph(
      languages.filter(language =>
        language.name.toLowerCase().includes(twitterSearchTermGraph.toLowerCase())
      )
    );
  }, [twitterSearchTermGraph, languages]);

  useEffect(() => {
    setFilteredTwitterWordCountLanguages(
      languages.filter(language =>
        language.name.toLowerCase().includes(twitterSearchTermWordCount.toLowerCase())
      )
    );
  }, [twitterSearchTermWordCount, languages]);



  useEffect(() => {
    setTwitterFilteredLanguagesWordCloud(
      languages.filter(language =>
        language.name.toLowerCase().includes(searchTwitterWordCloud.toLowerCase())
      )
    );
  }, [searchTwitterWordCloud, languages]);

  return (
    <div className="bg-gray-100 min-h-screen flex flex-col items-center p-4">
        <ChooseBar setSelectedYearAndMonths={setSelectedYearMonthsPage} setTheme={setTheme}   initialTheme={theme} />

      <div className="absolute top-40 left-24 flex justify-center">
  <div className="indicator bg-white rounded-md shadow-md max-w-[110px] text-center">
    <span className="indicator-item badge badge-secondary bg-accent border-accent border-1">THEME</span>
    <div className="mt-4 text-lg font-medium text-gray-600 break-words">
      <span className="text-accent">{theme}</span>
    </div>
  </div>
</div>

<div className="absolute top-72 left-20 flex justify-center">
  <div className="indicator bg-white p-2 rounded-md shadow-md max-w-[130px] text-center">
    <span className="indicator-item badge badge-secondary bg-accent border-accent border-1">TIMELINE</span>
    <div className="mt-4 text-lg font-medium text-gray-600 break-words">
    <span className="text-accent">{renderSelectedRange()}</span>
    </div>
  </div>
</div>




      <div className="flex flex-col md:flex-row w-full justify-center items-start md:space-y-0 md:space-x-4 mt-8 ml-24">
        {/* Card pour Reddit */}
        <div className="card bg-white rounded-lg shadow-lg p-4 flex-grow max-w-md">
          <h2 className="text-xl font-bold mb-2 text-center">Reddit</h2>
          <p className="text-sm mb-4 text-center">Know about sentiment on climate change, food safety etc all over the world</p>
          <div className="form-control bg-base-200 p-4 rounded-lg shadow-lg mb-4">
            <label className="cursor-pointer label">
              <span className="label-text">Sentiment Map</span>
              <input
                type="checkbox"
                checked={redditSentimentMap}
                onChange={(e) => setRedditSentimentMap(e.target.checked)}
                className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
              />
            </label>
          </div>
          <div className="form-control bg-base-200 p-4 rounded-lg shadow-lg mb-4">
            <label className="cursor-pointer label">
              <span className="label-text">Wordcloud</span>
              <input
                type="checkbox"
                checked={redditWordCloud}
                onChange={(e) => setRedditWorcloud(e.target.checked)}
                className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
              />
            </label>
          </div>
          {/* Collapse */}
          <div className={`collapse ${redditWordCloud ? 'block' : 'hidden'} mt-4 rounded-lg`} style={{ flex: 1 }}>
            <input type="checkbox" checked={redditWordCloud} readOnly className="hidden" />

            <div className="collapse-content p-4" style={{ maxHeight: '300px', overflowY: 'auto' }}>
              {/* Search Bar */}
              <input
                type="text"
                placeholder="Search for a country..."
                value={RedditSearchTermWordCloud}
                onChange={(e) => setRedditSearchTermWordCloud(e.target.value)}
                className="input input-bordered w-full mb-4"
              />
              {/* Country List */}
              {RedditSearchTermWordCloud === '' ? (
                <div className="py-1">Enter a search term to display countries</div>
              ) : (
                <ul>
                  {filteredRedditWordCloudCountries.length > 0 ? (
                    filteredRedditWordCloudCountries.map((country, index) => (
                      <li key={index} className="py-1 flex items-center">
                        <input
                          type="checkbox"
                          checked={selectedCountryWordCloudReddit === country.name}
                          onChange={() => setSelectedCountryWordCloudReddit(country.name)}
                          className="mr-2"
                        />
                        {country.name}
                      </li>
                    ))
                  ) : (
                    <li className="py-1">No countries found</li>
                  )}
                </ul>
              )}
            </div>
          </div>
        
          <div className="form-control bg-base-200 p-4 rounded-lg shadow-lg mb-4">


              <label className="cursor-pointer label">
                <span className="label-text">Circular Graph</span>
                <input
                  type="checkbox"
                  checked={redditCircularGraph}
                  onChange={(e) => setRedditCircularGraph(e.target.checked)}
                  className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
                />
              </label>
            </div>

            {/* Collapse */}
            <div className={`collapse ${redditCircularGraph ? 'block' : 'hidden'} mt-4 rounded-lg`} style={{ flex: 1 }}>
              <input type="checkbox" checked={redditCircularGraph} readOnly className="hidden" />

              <div className="collapse-content p-4" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {/* Search Bar */}
                <input
                  type="text"
                  placeholder="Search for a country..."
                  value={RedditSearchTermGraph}
                  onChange={(e) => setRedditSearchTermGraph(e.target.value)}
                  className="input input-bordered w-full mb-4"
                />
                {/* Country List */}
                {RedditSearchTermGraph === '' ? (
                  <div className="py-1">Enter a search term to display countries</div>
                ) : (
                  <ul>
                    {filteredRedditGraphCountries.length > 0 ? (
                      filteredRedditGraphCountries.map((country, index) => (
                        <li key={index} className="py-1 flex items-center">
                          <input
                            type="checkbox"
                            checked={selectedCountryRedditCircular === country.name}
                            onChange={() => setSelectedCountryRedditCircular(country.name)}
                            className="mr-2"
                          />
                          {country.name}
                        </li>
                      ))
                    ) : (
                      <li className="py-1">No countries found</li>
                    )}
                  </ul>
                )}
              </div>
            </div>
            
      

          <div className="bg-base-200 p-4 rounded-lg shadow-lg">
            <div className="form-control">
              <label className="cursor-pointer label">
                <span className="label-text">Word count sentiment</span>
                <input
                  type="checkbox"
                  checked={redditWordCount}
                  onChange={(e) => setRedditWordCount(e.target.checked)}
                  className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
                />
              </label>
            </div>

            {/* Collapse */}
            <div className={`collapse ${redditWordCount ? 'block' : 'hidden'} mt-4 rounded-lg`} style={{ flex: 1 }}>
              <input type="checkbox" checked={redditWordCount} readOnly className="hidden" />

              <div className="collapse-content p-4" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {/* Search Bar */}
                <input
                  type="text"
                  placeholder="Search for a country..."
                  value={searchRedditWordCount}
                  onChange={(e) => setSearchRedditWordCount(e.target.value)}
                  className="input input-bordered w-full mb-4"
                />
                {/* Country List */}
                {searchRedditWordCount === '' ? (
                  <div className="py-1">Enter a search term to display countries</div>
                ) : (
                  <ul>
                    {filteredRedditWordCountCountries.length > 0 ? (
                      filteredRedditWordCountCountries.map((country, index) => (
                        <li key={index} className="py-1 flex items-center">
                          <input
                            type="checkbox"
                            checked={selectedCountryWordCountReddit === country.name}
                            onChange={() => setSelectedCountryRedditWordCount(country.name)}
                            className="mr-2"
                          />
                          {country.name}
                        </li>
                      ))
                    ) : (
                      <li className="py-1">No countries found</li>
                    )}
                  </ul>
                )}
              </div>
            </div>
            
          </div>

        </div>
        

        {/* Card pour Twitter */}
        <div className="divider divider-horizontal">OR</div>
        <div className="card bg-white rounded-lg shadow-lg p-4 flex-grow max-w-md">
          <h2 className="text-xl font-bold mb-2 text-center">Twitter</h2>
          <p className="text-sm mb-4 text-center">Focus on one country or language and look at how spread is the sentiment</p>
          <div className="form-control bg-base-200 p-4 rounded-lg shadow-lg mb-4">
            <label className="cursor-pointer label">
              <span className="label-text">Sentiment Map</span>
              <input
                type="checkbox"
                checked={twitterSentimentMap}
                onChange={(e) => setTwitterSentimentMap(e.target.checked)}
                className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
              />
            </label>
          </div>
          <div className="form-control bg-base-200 p-4 rounded-lg shadow-lg mb-4">
            <label className="cursor-pointer label">
              <span className="label-text">Wordcloud</span>
              <input
                type="checkbox"
                checked={twitterWordCloud}
                onChange={(e) => setTwitterWordCloud(e.target.checked)}
                className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
              />
            </label>
          </div>
          {/* Collapse */}
          <div className={`collapse ${twitterWordCloud ? 'block' : 'hidden'} mt-4 rounded-lg`} style={{ flex: 1 }}>
            <input type="checkbox" checked={twitterWordCloud} readOnly className="hidden" />

            <div className="collapse-content p-4" style={{ maxHeight: '300px', overflowY: 'auto' }}>
              {/* Search Bar */}
              <input
                type="text"
                placeholder="Search for a language..."
                value={searchTwitterWordCloud}
                onChange={(e) => setSearchTwitterWordCloud(e.target.value)}
                className="input input-bordered w-full mb-4"
              />
              {/* Language List */}
              {searchTwitterWordCloud === '' ? (
                <div className="py-1">Enter a search term to display languages</div>
              ) : (
                <ul>
                  {twitterFilteredLanguagesWordCloud.length > 0 ? (
                    twitterFilteredLanguagesWordCloud.map((language) => (
                      <li key={language.code} className="py-1 flex items-center">
                        <input
                          type="checkbox"
                          checked={selectedTwitterLanguageWordCloud === language.code}
                          onChange={() => handleLanguageChangeWordCloud(language.code)}
                          className="mr-2"
                        />
                        {language.name}
                      </li>
                    ))
                  ) : (
                    <li className="py-1">No languages found</li>
                  )}
                </ul>
              )}
            </div>
          </div>

          <div className="bg-base-200 p-4 rounded-lg shadow-lg mb-4">
            <div className="form-control">
              <label className="cursor-pointer label">
                <span className="label-text">Circular Graph</span>
                <input
                  type="checkbox"
                  checked={twitterCircularGraph}
                  onChange={(e) => setTwitterCircularGraph(e.target.checked)}
                  className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
                />
              </label>
            </div>

            {/* Collapse */}
            <div className={`collapse ${twitterCircularGraph ? 'block' : 'hidden'} mt-4 rounded-lg`} style={{ flex: 1 }}>
              <input type="checkbox" checked={twitterCircularGraph} readOnly className="hidden" />

              <div className="collapse-content p-4" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {/* Search Bar */}
                <input
                  type="text"
                  placeholder="Search for a language..."
                  value={twitterSearchTermGraph}
                  onChange={(e) => setTwitterSearchTermGraph(e.target.value)}
                  className="input input-bordered w-full mb-4"
                />
                {/* Country List */}
                {twitterSearchTermGraph === '' ? (
                  <div className="py-1">Enter a search term to display languages</div>
                ) : (
                  <ul>
                    {twitterFilteredLanguagesGraph.length > 0 ? (
                      twitterFilteredLanguagesGraph.map((language) => (
                        <li key={language.code} className="py-1 flex items-center">
                          <input
                            type="checkbox"
                            checked={selectedTwitterLanguageGraph === language.code}
                            onChange={() => handleLanguageChangeGraph(language.code)}
                            className="mr-2"
                          />
                          {language.name}
                        </li>
                      ))
                    ) : (
                      <li className="py-1">No languages found</li>
                    )}
                  </ul>
                )}
              </div>
            </div>
          </div>
        
        <div className="bg-base-200 p-4 rounded-lg shadow-lg">
            <div className="form-control">
              <label className="cursor-pointer label">
                <span className="label-text">Word count sentiment</span>
                <input
                  type="checkbox"
                  checked={twitterWordCount}
                  onChange={(e) => setTwitterWordCount(e.target.checked)}
                  className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
                />
              </label>
            </div>

            {/* Collapse */}
            <div className={`collapse ${twitterWordCount ? 'block' : 'hidden'} mt-4 rounded-lg`} style={{ flex: 1 }}>
              <input type="checkbox" checked={twitterWordCount} readOnly className="hidden" />

              <div className="collapse-content p-4" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {/* Search Bar */}
                <input
                  type="text"
                  placeholder="Search for a country..."
                  value={twitterSearchTermWordCount}
                  onChange={(e) => setTwitterSearchTermWordCount(e.target.value)}
                  className="input input-bordered w-full mb-4"
                />
                {/* Country List */}
                {twitterSearchTermWordCount === '' ? (
                  <div className="py-1">Enter a search term to display countries</div>
                ) : (
                  <ul>
                    {filteredTwitterWordCountLanguages.length > 0 ? (
                      filteredTwitterWordCountLanguages.map((language, index) => (
                        <li key={index} className="py-1 flex items-center">
                          <input
                            type="checkbox"
                            checked={selectedLanguageTwitterWordCount === language.code}
                            onChange={() => setSelectedLanguageTwitterWordCount(language.code)}
                            className="mr-2"
                          />
                          {language.name}
                        </li>
                      ))
                    ) : (
                      <li className="py-1">No countries found</li>
                    )}
                  </ul>
                )}
              </div>
            </div>
            
          </div>

        
      </div>
      </div>

      <div className="grid gap-4 px-10 mb-8 mt-8 w-full">
        {/* Range slider for sentiment evolution */}
        <div className="col-span-3 bg-white rounded-lg p-4 mt-4 shadow-lg">
          <div className="flex flex-row items-center justify-between">
            <h2 className="text-xl font-bold mr-4">Sentiment Evolution (2008-2024)</h2>
            <input
              type="checkbox"
              checked={sentimentEvolution}
              onChange={(e) => handleSentimentChange(e.target.checked)}
              className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
            />
          </div>
          <div className={`collapse ${sentimentEvolution ? 'block' : 'hidden'} rounded-lg`} style={{ flex: 1 }}>
            <input type="checkbox" checked={sentimentEvolution} readOnly className="hidden" />
            <div className="collapse-content p-4" style={{ maxHeight: '200px', overflowY: 'auto' }}>
              <div className="flex flex-row items-center mb-4 space-x-4 justify-evenly">
                <div className="flex flex-col space-y-4">
                  <div className="flex flex-row items-center">
                    <input
                      type="checkbox"
                      checked={twitterEvolutionChecked}
                      onChange={(e) => setTwitterEvolutionChecked(e.target.checked)}
                      className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
                    />
                    <label className="ml-2 text-lg">Twitter</label>
                  </div>
                  <div className="flex flex-row items-center">
                    <input
                      type="checkbox"
                      checked={redditEvolutionChecked}
                      onChange={(e) => handleRedditEvolutionClick(e.target.checked)}
                      className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
                    />
                    <label className="ml-2 text-lg">Reddit</label>
                  </div>
                  <div className={`collapse ${redditEvolutionChecked ? 'block' : 'hidden'} mt-4 rounded-lg`} style={{ flex: 1 }}>
                    <input type="checkbox" checked={redditEvolutionChecked} readOnly className="hidden" />
                    <div className="collapse-content p-4" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                      {/* Search Bar */}
                      <input
                        type="text"
                        placeholder="Search for a country..."
                        value={evolutionSearchTerm}
                        onChange={(e) => setEvolutionSearchTerm(e.target.value)}
                        className="input input-bordered w-full mb-4"
                      />
                      {/* Country List */}
                      {evolutionSearchTerm === '' ? (
                        <div className="py-1">Enter a search term to display countries</div>
                      ) : (
                        <ul>
                          {filteredCountriesEvolution.length > 0 ? (
                            filteredCountriesEvolution.map((country, index) => (
                              <li key={index} className="py-1 flex items-center">
                                <input
                                  type="checkbox"
                                  checked={selectedCountryEvolution === country.name}
                                  onChange={() => setSelectedCountryEvolution(country.name)}
                                  className="mr-2"
                                />
                                {country.name}
                              </li>
                            ))
                          ) : (
                            <li className="py-1">No countries found</li>
                          )}
                        </ul>
                      )}
                    </div>
                  </div>
                </div>
                <RangeSlider onRangeChange={handleRangeChangeEvolution} />
              </div>
            </div>
          </div>
        </div>
      </div>
     
      <div className="grid gap-4 px-10 w-full">
        {/* Range slider for sentiment evolution */}
        <div className="col-span-3 bg-white rounded-lg p-4 shadow-lg">
          <div className="flex flex-row items-center justify-between">
            <h2 className="text-xl font-bold mr-4">Posts evolution (2008-2024)</h2>
            <input
              type="checkbox"
              checked={postEvolution}
              onChange={(e) => handlePostEvolutionChange(e.target.checked)}
              className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
            />
          </div>
          <div className={`collapse ${postEvolution ? 'block' : 'hidden'} rounded-lg`} style={{ flex: 1 }}>
            <input type="checkbox" checked={postEvolution} readOnly className="hidden" />
            <div className="collapse-content p-4" style={{ maxHeight: '200px', overflowY: 'auto' }}>
              <div className="flex flex-row items-center mb-4 space-x-4 justify-evenly">
                <div className="flex flex-col space-y-4">
                 
                  <div className="flex flex-row items-center">
                    <input
                      type="checkbox"
                      checked={redditPostEvolutionChecked}
                      onChange={(e) => handleRedditPostEvolutionClick(e.target.checked)}
                      className="checkbox border-accent [--chkbg:theme(colors.accent)] [--chkfg:white]"
                    />
                    <label className="ml-2 text-lg">Reddit</label>
                  </div>
                  <div className={`collapse ${redditPostEvolutionChecked ? 'block' : 'hidden'} mt-4 rounded-lg`} style={{ flex: 1 }}>
                    <input type="checkbox" checked={redditPostEvolutionChecked} readOnly className="hidden" />
                    <div className="collapse-content p-4" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                      {/* Search Bar */}
                      <input
                        type="text"
                        placeholder="Search for a country..."
                        value={postEvolutionSearchTerm}
                        onChange={(e) => setPostEvolutionSearchTerm(e.target.value)}
                        className="input input-bordered w-full mb-4"
                      />
                      {/* Country List */}
                      {postEvolutionSearchTerm === '' ? (
                        <div className="py-1">Enter a search term to display countries</div>
                      ) : (
                        <ul>
                          {filteredCountriesPostEvolution.length > 0 ? (
                            filteredCountriesPostEvolution.map((country, index) => (
                              <li key={index} className="py-1 flex items-center">
                                <input
                                  type="checkbox"
                                  checked={selectedPostEvolutionCountry === country.name}
                                  onChange={() => setSelectedPostEvolutionCountry(country.name)}
                                  className="mr-2"
                                />
                                {country.name}
                              </li>
                            ))
                          ) : (
                            <li className="py-1">No countries found</li>
                          )}
                        </ul>
                      )}
                    </div>
                  </div>
                </div>
                <RangeSlider onRangeChange={handleRangeChangePostEvolution} />
              </div>
            </div>
          </div>
        </div>
      </div>


      <div className="flex justify-center items-center mt-4 mb-4">
        <button
          onClick={runScript}
          className="btn btn-ghost rounded-btn px-8 py-2 border border-gray-300 bg-white shadow-md hover:bg-accent hover:text-white focus:text-white focus:bg-accent text-xl mb-16"
        >



          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>
    </div>
  );
}

