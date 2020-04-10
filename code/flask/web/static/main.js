const add_results = (results, tpSearchResult, divResults) => {
  for (result of results) {
    // Clone template
    const searchResult = tpSearchResult.content.cloneNode(true);

    // Query and update content
    searchResult.querySelector(".search-result-title").innerHTML = result.title;
    searchResult.querySelector(
      ".search-result-subtitle"
    ).innerHTML = `${result.authors}, ${result.source}, ${result.journal}, ${result.publish_time}`;
    searchResult.querySelector(
      ".search-result-text-snippet"
    ).innerHTML = `${result.body_text.substring(0, 300)} {...}`;
    searchResult.querySelector(".search-result-open-url").href = result.url;

    divResults.appendChild(searchResult);
  }
};

const load_more_visibility = (btnLoadMore, is_bottom) => {
  if (is_bottom) {
    btnLoadMore.classList.add("hidden");
  } else {
    btnLoadMore.classList.remove("hidden");
  }
};

window.onload = function () {
  // Get DOM elements
  const txtSearchBox = document.querySelector("#search-box");
  const divResults = document.querySelector("#results");
  const btnLoadMore = document.querySelector("#load-more");
  const tpSearchResult = document.querySelector("#template-search-result");

  const urlParams = new URLSearchParams(window.location.search);
  const query = urlParams.get("q") || "";
  txtSearchBox.value = query;

  // Perform search
  let num_per_load;
  let num_results = 0;
  if (query.length > 0) {
    fetch(`/search?q=${query}`, { method: "POST" })
      .then((res) => res.json())
      .then((json) => {
        // Add results and set "Load more" button visibility
        add_results(json.results, tpSearchResult, divResults);
        num_results += json.results.length;
        load_more_visibility(btnLoadMore, json.total_results == num_results);
        num_per_load = json.num_per_load;
      })
      .catch((err) => console.log("Error while searching: ", err));
  }

  // Load more button onclick event
  btnLoadMore.addEventListener("click", function () {
    fetch(
      `/search?q=${query}&start=${num_results}&stop=${
        num_results + num_per_load
      }`,
      { method: "POST" }
    )
      .then((res) => res.json())
      .then((json) => {
        // Add results and set "Load more" button visibility
        add_results(json.results, tpSearchResult, divResults);
        num_results += json.results.length;
        load_more_visibility(btnLoadMore, json.total_results == num_results);
        num_per_load = json.num_per_load;
      })
      .catch((err) => console.log("Error while loading more: ", err));
  });
};
