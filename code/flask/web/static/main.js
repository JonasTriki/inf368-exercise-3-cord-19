const add_results = (results, tpSearchResult, divResults, bodyTextModal) => {
  for (const result of results) {
    // Clone template
    const searchResult = tpSearchResult.content.cloneNode(true);

    // Query DOM elements
    const searchTitle = searchResult.querySelector(".search-result-title");
    const searchSubtitle = searchResult.querySelector(
      ".search-result-subtitle"
    );
    const searchTextSnippet = searchResult.querySelector(
      ".search-result-text-snippet"
    );

    // Update content
    searchTitle.innerHTML = result.title;
    searchTitle.href = result.url;
    const subtitle = `${result.authors}, ${result.source}, ${result.journal}, ${result.publish_time}`;
    searchSubtitle.innerHTML = subtitle;
    searchTextSnippet.innerHTML = `${result.body_text.substring(0, 300)} {...}`;
    searchTextSnippet.addEventListener("click", () => {
      // Set title/subtitle
      bodyTextModal.querySelector(".modal-title").innerHTML = result.title;
      bodyTextModal.querySelector(".modal-subtitle").innerHTML = subtitle;

      // Add paragraphs for each newline in the body text
      modalTextDiv = bodyTextModal.querySelector(".modal-text");
      while (modalTextDiv.firstChild) {
        modalTextDiv.firstChild.remove();
      }
      for (const sentence of result.body_text.split("\n")) {
        const sentPara = document.createElement("p");
        sentPara.innerHTML = sentence;
        modalTextDiv.appendChild(sentPara);
      }
      bodyTextModal.style.display = "flex";
    });

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
  const bodyTextModal = document.querySelector("#cord-body-text-modal");

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
        add_results(json.results, tpSearchResult, divResults, bodyTextModal);
        num_results += json.results.length;
        load_more_visibility(btnLoadMore, json.total_results == num_results);
        num_per_load = json.num_per_load;
      })
      .catch((err) => console.log("Error while searching: ", err));
  }

  // Modal events
  bodyTextModal.querySelector(".modal-close").addEventListener("click", () => {
    bodyTextModal.style.display = "none";
  });
  window.onclick = function (e) {
    if (e.target == bodyTextModal) {
      bodyTextModal.style.display = "none";
    }
  };
  window.addEventListener('keydown', function (e) {
    if (
      (e.key == "Escape" || e.key == "Esc" || e.keyCode == 27) &&
      e.target.nodeName == "BODY"
    ) {
      bodyTextModal.style.display = "none";
      e.preventDefault();
      return false;
    }
  }, true);

  // Load more button onclick event
  if (btnLoadMore !== null) {
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
          add_results(json.results, tpSearchResult, divResults, bodyTextModal);
          num_results += json.results.length;
          load_more_visibility(btnLoadMore, json.total_results == num_results);
          num_per_load = json.num_per_load;
        })
        .catch((err) => console.log("Error while loading more: ", err));
    });
  }
};
