"use client";
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

interface ChooseBarProps {
  setSelectedYearAndMonths: (yearsMonths: { year: string; month: string }[]) => void;
  setTheme: (theme: string) => void;
  initialTheme: string | null;
}

export const ChooseBar: React.FC<ChooseBarProps> = ({ setSelectedYearAndMonths, setTheme, initialTheme }) => {
  const router = useRouter();
  const [selectedYearMonths, setSelectedYearMonths] = useState<{ year: string; month: string }[]>([]);
  const [selectedTheme, setSelectedTheme] = useState<string | null>(initialTheme);
  const [hoveredYear, setHoveredYear] = useState<string | null>(null);
  const [activeYear, setActiveYear] = useState<string | null>(null); // State to track the active year
  const [showMonths, setShowMonths] = useState(false); // State to control month visibility

  const years = Array.from({ length: 2023 - 2008 + 1 }, (_, i) => 2008 + i);
  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];
  const [startYearIndex, setStartYearIndex] = useState(0);
  const displayedYears = years.slice(startYearIndex, startYearIndex + 5);

  const updateUrl = (yearMonths: { year: string; month: string }[], theme: string | null) => {
    const yearQuery = yearMonths.map(ym => `${ym.year}:${ym.month}`).join('|');
    const themeQuery = theme ? `&theme=${theme}` : '';
    router.push(`/home?years=${yearQuery}${themeQuery}`);
  };

  const handleMonthClick = (event: React.MouseEvent<HTMLButtonElement>, year: string, month: string) => {
    event.stopPropagation(); // Empêche la propagation de l'événement
    

    // Si le mois est déjà sélectionné, le supprimer
    if (selectedYearMonths.some(ym => ym.year === year && ym.month === month)) {
        const updatedYearMonths = selectedYearMonths.filter(ym => !(ym.year === year && ym.month === month));
        setSelectedYearMonths(updatedYearMonths);
        setSelectedYearAndMonths(updatedYearMonths);
        updateUrl(updatedYearMonths, selectedTheme);
        return;
    }

    // Si l'année est déjà sélectionnée mais avec un mois différent, ajouter le nouveau mois
    if (selectedYearMonths.some(ym => ym.year === year)) {
        // Ajouter un nouvel élément avec le même année mais un mois différent
        const updatedYearMonths = [...selectedYearMonths, { year, month }];

        // Remove the first element if the length exceeds 2
        if (updatedYearMonths.length > 2) {
            updatedYearMonths.splice(0, 1); // Supprime le premier élément pour garder une longueur de 2
        }

        setSelectedYearMonths(updatedYearMonths);
        setSelectedYearAndMonths(updatedYearMonths);
        updateUrl(updatedYearMonths, selectedTheme);
        return;
    }

    // Si l'année n'est pas sélectionnée, ajouter le nouvel élément
    const updatedYearMonths = [...selectedYearMonths, { year, month }];

    // Remove the first element if the length exceeds 2
    if (updatedYearMonths.length > 2) {
        updatedYearMonths.splice(0, 1); // Supprime le premier élément pour garder une longueur de 2
    }

    setSelectedYearMonths(updatedYearMonths);
    setSelectedYearAndMonths(updatedYearMonths);
    setHoveredYear(null);
    setShowMonths(false); // Masquer les mois après en avoir sélectionné un
    updateUrl(updatedYearMonths, selectedTheme);
};


  
  
  

  const handleYearClick = (year: string) => {
   
    if (selectedYearMonths.some(ym => ym.year === year)) {
      // If the year is already selected, remove it
      const updatedYearMonths = selectedYearMonths.filter(ym => ym.year !== year);
      setSelectedYearMonths(updatedYearMonths);
      setSelectedYearAndMonths(updatedYearMonths);
      updateUrl(updatedYearMonths, selectedTheme);
    }
  };

  const handleThemeClick = (theme: string) => {
    setTheme(theme);
    setSelectedTheme(theme);
    updateUrl(selectedYearMonths, theme);
  };

  const handleNext = () => {
    if (startYearIndex + 5 < years.length) {
      setStartYearIndex(startYearIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (startYearIndex > 0) {
      setStartYearIndex(startYearIndex - 1);
    }
  };

  return (
    <div className="navbar bg-white rounded-lg p-4">
      <div className="navbar-start flex items-center">
        <Image src="/GhentUniversityLogo.png" width={300} height={60} alt="Ghent University Logo" />
      </div>

      <div className="relative max-w-[500px] mx-auto mt-4">
        <div className="navbar-center flex items-center justify-center relative">
          <button onClick={handlePrevious} className="absolute left-0 z-10 flex items-center">
            <img src="/arrowLeft.svg" alt="Previous" width="24" height="24" />
          </button>
          <ul className="menu menu-horizontal px-1 text-xl flex space-x-6 whitespace-nowrap mx-4">
            {displayedYears.map((year) => (
              <li
                key={year}
                className="relative"
                onClick={() => handleYearClick(year.toString())}
                onMouseEnter={() => {
                  setHoveredYear(year.toString());
                  setShowMonths(true); 
                }}
                onMouseLeave={() => {
                 
                    setHoveredYear( null);
                    setShowMonths(false); // Hide months when leaving if not active
                  
                }}
              >
                <span
                  className={`${
                    selectedYearMonths.some(ym => ym.year === year.toString()) ? 'text-accent' : ''
                  } cursor-pointer`}
                >
                  {year}
                </span>

                {showMonths && (activeYear === year.toString() || hoveredYear === year.toString()) && (
      <div className="absolute left-1/2 transform -translate-x-1/2 mt-11 bg-white border rounded shadow-lg p-2">
        {months.map((month) => (
          <button
            key={`${year}-${month}`}
            onClick={(event) => handleMonthClick(event, year.toString(), month)}
            className={`btn btn-sm m-1 ${
              selectedYearMonths.find(ym => ym.year === year.toString() && ym.month === month)
                ? 'bg-accent text-white' // Mois sélectionné
                : 'bg-gray-200 text-black' // Mois non sélectionné
            }`}
          >
            {month}
          </button>
                  
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
          <button onClick={handleNext} className="absolute right-0 z-10 flex items-center">
            <img src="/arrowRight.svg" alt="Next" width="24" height="24" />
          </button>
        </div>
      </div>

      <div className="navbar-end flex items-center">
        <div className="dropdown dropdown-end">
          <div tabIndex={0} role="button" className="btn btn-ghost rounded-btn px-8 py-3 border border-gray-300 bg-white shadow-md hover:bg-accent hover:text-white focus:text-white focus:bg-accent">
            THEME
          </div>
          <ul
            tabIndex={0}
            className="menu dropdown-content bg-base-100 rounded-box z-[1] mt-4 w-52 p-2 shadow-lg border border-gray-300">
            {['Climate change', 'Micro Plastic'].map((theme) => (
              <li key={theme}>
                <a
                  onClick={() => handleThemeClick(theme)}
                  className={`${
                    selectedTheme === theme ? 'bg-accent text-white' : ''
                  } hover:bg-accent focus:bg-accent hover:text-white focus:text-white`}
                >
                  {theme}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ChooseBar;
