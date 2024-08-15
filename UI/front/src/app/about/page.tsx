import React from 'react';

const AboutPage = () => {
  return (
    <div className="bg-grey-100 p-4">
      {/* Premier Rectangle */}
      <div className="flex items-center justify-center mt-12 mb-12">
        <div className="bg-white border-accent border-2 p-6 rounded-lg shadow-lg max-w-3xl w-full">
          <h1 className="text-4xl font-bold mb-6 text-center">About This Project</h1>
          <p className="text-lg mb-4 leading-relaxed">
            This project, developed by Théo Hinaut, a student at ISEN in Lille, in collaboration with Maxime Van Haeverbeke and Mrs. Christine Hung from Ghent University (Faculty of Bioscience Engineering), aims to analyze worldwide social media data to gauge the general sentiment on topics like climate change and food engagement.
          </p>
          <p className="text-lg mb-4 leading-relaxed">
            Through advanced data processing and sentiment analysis techniques, we strive to understand public opinions and trends on important global issues that affect our planet.
          </p>
          <p className="text-lg leading-relaxed">
            We would like to extend our heartfelt thanks to everyone who has supported us throughout this journey, including our mentors, colleagues, and the broader community.
          </p>
        </div>
      </div>

      {/* Autres Rectangles côte à côte */}
      <div className="flex flex-wrap justify-center space-x-6 space-y-6 mb-24">
        <div className="bg-white border-accent border-2 p-6 rounded-lg shadow-lg w-full md:w-1/2 lg:w-1/3">
          <h2 className="text-3xl font-bold mb-4">Sentiment Map</h2>
          <p className="text-lg leading-relaxed">
            The Sentiment Map visually represents the geographic distribution of sentiments regarding various topics. By analyzing social media data, we can identify regions with positive or negative sentiments, providing insights into public opinion across different locations.
          </p>
        </div>

        <div className="bg-white border-accent border-2 p-6 rounded-lg shadow-lg w-full md:w-1/2 lg:w-1/3">
          <h2 className="text-3xl font-bold mb-4">Wordcloud</h2>
          <p className="text-lg leading-relaxed">
            A Wordcloud is a visual representation of the most frequently used words in a dataset. In this project, it helps to highlight key terms related to climate change and food engagement, allowing us to quickly grasp the main topics of discussion among users.
          </p>
        </div>

        <div className="bg-white border-accent border-2 p-6 rounded-lg shadow-lg w-full md:w-1/2 lg:w-1/3">
          <h2 className="text-3xl font-bold mb-4">Circular Graph</h2>
          <p className="text-lg leading-relaxed">
            The Circular Graph, or pie chart, illustrates the proportion of different sentiments expressed in the social media posts. It provides a quick overview of how opinions are distributed across positive, negative, and neutral sentiments, aiding in the understanding of public sentiment.
          </p>
        </div>

        <div className="bg-white border-accent border-2 p-6 rounded-lg shadow-lg w-full md:w-1/2 lg:w-1/3">
          <h2 className="text-3xl font-bold mb-4">Word Count Sentiment</h2>
          <p className="text-lg leading-relaxed">
            The Word Count Sentiment analysis quantifies the sentiment of words used in the posts. By examining the frequency and sentiment associated with specific words, we can identify which terms resonate most with the audience and gauge overall sentiment towards various topics.
          </p>
        </div>

        <div className="bg-white border-accent border-2 p-6 rounded-lg shadow-lg w-full md:w-1/2 lg:w-1/3">
          <h2 className="text-3xl font-bold mb-4">Sentiment Evolution (2008-2024)</h2>
          <p className="text-lg leading-relaxed">
            The Sentiment Evolution chart tracks changes in sentiment over the years, from 2008 to 2024. By analyzing trends over time, we can observe how public sentiment towards climate change and food issues has evolved, highlighting significant events or shifts in opinion.
          </p>
        </div>

        <div className="bg-white border-accent border-2 p-6 rounded-lg shadow-lg w-full md:w-1/2 lg:w-1/3">
          <h2 className="text-3xl font-bold mb-4">Posts Evolution (2008-2024)</h2>
          <p className="text-lg leading-relaxed">
            The Posts Evolution graph illustrates the growth in the number of social media posts related to climate change and food engagement from 2008 to 2024. This visualization helps us understand how interest and engagement with these topics have changed over time.
          </p>
        </div>
      </div>
    </div>
  );
}

export default AboutPage;
