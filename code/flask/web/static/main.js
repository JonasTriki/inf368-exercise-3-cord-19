const add_results = (results, current_num_results, tpSearchResult, divResults, bodyTextModal) => {
  for (let i = 0; i < results.length; i++) {
    result = results[i];

    // Clone template
    const searchResult = tpSearchResult.content.cloneNode(true);

    // Query DOM elements
    const searchTitleATag = searchResult.querySelector(".search-result-title a");
    const searchTitleNumber = searchResult.querySelector(".search-result-title .search-result-number");
    const searchSubtitle = searchResult.querySelector(
      ".search-result-subtitle"
    );
    const searchTextSnippet = searchResult.querySelector(
      ".search-result-text-snippet"
    );

    // Update content
    searchTitleNumber.innerHTML = `${current_num_results + i + 1}.`;
    const title = result.title || 'No title';
    searchTitleATag.innerHTML = title;
    searchTitleATag.href = result.url;
    const subtitle = `${result.authors}, ${result.source}, ${result.journal}, ${result.publish_time}`;
    searchSubtitle.innerHTML = subtitle;
    searchTextSnippet.innerHTML = `${result.body_text.substring(0, 300)} {...}`;
    searchTextSnippet.addEventListener("click", () => {
      // Set title/subtitle
      bodyTextModal.querySelector(".modal-title").innerHTML = title;
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

const load_more_visibility = (btnLoadMore, total_results, current_num_results) => {
  if (total_results == 0 || total_results == current_num_results) {
    btnLoadMore.classList.add("hidden");
  } else {
    btnLoadMore.classList.remove("hidden");
  }
};

const no_results_found = (divNoResults, total_results) => {
  if (total_results > 0) {
    divNoResults.classList.add("hidden");
  } else {
    divNoResults.classList.remove("hidden");
  }
}

const set_loading_visibility = (visible, divLoading) => {
  if (!visible) {
    divLoading.classList.add("hidden");
  } else {
    divLoading.classList.remove("hidden");
  }
}

window.onload = function () {
  // Get DOM elements
  const txtSearchBox = document.querySelector("#search-box");
  const divResults = document.querySelector("#results");
  const divLoading = document.querySelector("#loading");
  const divNoResults = document.querySelector("#no-results");
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
    set_loading_visibility(true, divLoading);
    fetch(`/search?q=${query}`, { method: "POST" })
      .then((res) => res.json())
      .then((json) => {
        if (json.error_msgs) {
          console.log("error, ", json.error_msgs);
          return;
        }
        const results = JSON.parse(json.results);
        
        // Add results and set "Load more" button visibility
        add_results(results, num_results, tpSearchResult, divResults, bodyTextModal);
        num_results += results.length;
        load_more_visibility(btnLoadMore, json.total_results, num_results);
        no_results_found(divNoResults, json.total_results);
        num_per_load = json.num_per_load;
        set_loading_visibility(false, divLoading);
      })
      .catch((err) => {
        console.log("Error while searching: ", err);
        set_loading_visibility(false, divLoading);
      });
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
  window.addEventListener(
    "keydown",
    function (e) {
      if (
        (e.key == "Escape" || e.key == "Esc" || e.keyCode == 27) &&
        e.target.nodeName == "BODY"
      ) {
        bodyTextModal.style.display = "none";
        e.preventDefault();
        return false;
      }
    },
    true
  );

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
          if (json.error_msgs) {
            console.log("error, ", json.error_msgs);
            return;
          }
          const results = JSON.parse(json.results);

          // Add results and set "Load more" button visibility
          add_results(results, num_results, tpSearchResult, divResults, bodyTextModal);
          num_results += results.length;
          load_more_visibility(btnLoadMore, json.total_results, num_results);
          num_per_load = json.num_per_load;
        })
        .catch((err) => console.log("Error while loading more: ", err));
    });
  }
};
