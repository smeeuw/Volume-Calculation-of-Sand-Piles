// Map Settings
const mapOpacity = 1.0;
const mapResolution = 256;

// Sidebar Settings - used to open the first panel of the sidebar
const sidebarDivName = "sidebar";
const sidebarHomeHrefName = "home";

// Polygon Settings
const distinctColors = [
  "#e6194B",
  "#3cb44b",
  "#ffe119",
  "#4363d8",
  "#f58231",
  "#911eb4",
  "#42d4f4",
  "#f032e6",
  "#bfef45",
  "#fabed4",
  "#469990",
  "#dcbeff",
  "#9A6324",
  "#fffac8",
  "#800000",
  "#aaffc3",
  "#808000",
  "#ffd8b1",
  "#000075",
  "#a9a9a9",
];

const polygonFillOpacity = 0.3;
const polygonBorderOpacity = 0.6;
const polygonBorderColor = "#000";

// Marker settings
const markerBGColor = "black";

// Progress Bar Settings
const expectedTimeInSeconds = 120;

//JQuery
const progressBarContainerElement = $("#progress-bar-container");
const progressBarElement = $("#progress-bar");
const progressStatusElement = $("#progress-status");
const mapElement = $("#map");
const sidebarElement = $("#sidebar");
const sidebarListElement = $("#polygon-list");

// Routing
const volumeCalculationRoute = "/mapWithPolygons";
const polygonRoute = "/polygons";
const orthophotoRoute = "/orthophoto";

/**
 * Initializes the page by setting up the progress bar, hiding the map, and requesting the server to start the calculation
 * of the volume.
 * @param {Object} progressBarElement - The jQuery object for the progress bar element.
 * @param {Object} progressStatusElement - The jQuery object for the progress status element.
 * @param {number} expectedTimeInSeconds - The expected time in seconds for the progress bar to complete.
 * @param {Object} mapElement - The jQuery object for the map element.
 * @param {Object} progressBarContainerElement - The jQuery object for the progress bar container element.
 * @param {string} volumeCalculationRoute - The route to fetch volume calculation data.
 * @param {string} polygonRoute - The route to fetch polygon data.
 * @param {string} orthophotoRoute - The route to fetch orthophoto data.
 * @param {Array} distinctColors - An array of distinct colors for polygons.
 * @param {string} polygonBorderColor - The color for polygon borders.
 * @param {number} polygonBorderOpacity - The opacity for polygon borders.
 * @param {number} polygonFillOpacity - The opacity for polygon fills.
 * @param {string} markerBGColor - The background color of the marker.
 * @param {number} mapOpacity - The opacity for map layers.
 * @param {number} mapResolution - The resolution for map tiles.
 * @param {string} sidebarDivName - The name of the sidebar div.
 * @param {string} sidebarHomeHrefName - The name of the href for the sidebar home. Used to open it on loading.
 * @param {Object} sidebarElement - The JQuery object for the sidebar.
 * @param {Object} sidebarListElement - The JQuery object of the volume list for the sidebar.
 */
function initializePage(
  progressBarElement,
  progressStatusElement,
  expectedTimeInSeconds,
  mapElement,
  progressBarContainerElement,
  volumeCalculationRoute,
  polygonRoute,
  orthophotoRoute,
  distinctColors,
  polygonBorderColor,
  polygonBorderOpacity,
  polygonFillOpacity,
  markerBGColor,
  mapOpacity,
  mapResolution,
  sidebarDivName,
  sidebarHomeHrefName,
  sidebarElement,
  sidebarListElement
) {
  setupProgressBar(
    progressBarElement,
    progressStatusElement,
    expectedTimeInSeconds
  );

  showProgressBarHideMap(
    mapElement,
    progressBarContainerElement,
    sidebarElement
  );

  fetch(volumeCalculationRoute)
    .then(function (response) {
      if (!response.ok) {
        throw new Error("Could not receive polygons from server");
      }
      initMap(
        polygonRoute,
        orthophotoRoute,
        distinctColors,
        polygonBorderColor,
        polygonBorderOpacity,
        polygonFillOpacity,
        markerBGColor,
        mapOpacity,
        mapResolution,
        mapElement,
        progressBarContainerElement,
        sidebarDivName,
        sidebarHomeHrefName,
        sidebarElement,
        sidebarListElement
      );
    })
    .catch(function (error) {
      console.error("Error when calculating the volume:", error);
    });
}

/**
 * Initializes the map by fetching the resulting polygons and orthophoto the orthophoto from the server,
 * and adding them to the map.
 * @param {string} polygonRoute - The route to fetch polygon data.
 * @param {string} orthophotoRoute - The route to fetch orthophoto data.
 * @param {Array} distinctColors - An array of distinct colors for polygons.
 * @param {string} polygonBorderColor - The color for polygon borders.
 * @param {number} polygonBorderOpacity - The opacity for polygon borders.
 * @param {number} polygonFillOpacity - The opacity for polygon fills.
 * @param {string} markerBGColor - The background color of the marker.
 * @param {number} mapOpacity - The opacity for map layers.
 * @param {number} mapResolution - The resolution for map tiles.
 * @param {Object} mapElement - The jQuery object for the map element.
 * @param {Object} progressBarContainerElement - The jQuery object for the progress bar container element.
 * @param {string} sidebarDivName - The name of the sidebar div.
 * @param {string} sidebarHomeHrefName - The name of the href for the sidebar home. Used to open it on loading.
 * @param {Object} sidebarElement - The JQuery object for the sidebar.
 * @param {Object} sidebarListElement - The JQuery object of the volume list for the sidebar.
 */
async function initMap(
  polygonRoute,
  orthophotoRoute,
  distinctColors,
  polygonBorderColor,
  polygonBorderOpacity,
  polygonFillOpacity,
  markerBGColor,
  mapOpacity,
  mapResolution,
  mapElement,
  progressBarContainerElement,
  sidebarDivName,
  sidebarHomeHrefName,
  sidebarElement,
  sidebarListElement
) {
  try {
    const polygonData = await fetchPolygons(polygonRoute);
    const orthophotoData = await fetchOrthophoto(orthophotoRoute);

    hideProgressBarShowMap(
      mapElement,
      progressBarContainerElement,
      sidebarElement
    );

    const map = L.map("map").setView([0, 0], 5);

    addSidebarToMap(map, sidebarDivName, sidebarHomeHrefName);

    addOSMBasemapToMap(map);

    addPolygonsToMap(
      map,
      polygonData,
      distinctColors,
      polygonBorderColor,
      polygonBorderOpacity,
      polygonFillOpacity,
      markerBGColor,
      sidebarListElement
    );

    const orthoLayer = addOrthophotoToMap(
      map,
      orthophotoData,
      mapOpacity,
      mapResolution
    );
    map.fitBounds(orthoLayer.getBounds());
  } catch (error) {
    console.error("Error initializing map: ", error);
    hideProgressBarShowMap(
      mapElement,
      progressBarContainerElement,
      sidebarElement
    );
  }
}

/**
 * Fetches the polygons with volume from the server.
 * @param {string} polygonRoute - The route to fetch polygon data.
 * @returns {Promise<Array>} An array of polygon GeoJSON data from the server.
 * @throws Will throw an error if fetching the polygons fails.
 */
async function fetchPolygons(polygonRoute) {
  try {
    const response = await fetch(polygonRoute);
    const zipBlob = await response.blob();
    const zip = await JSZip.loadAsync(zipBlob);

    const polygonStrings = [];
    await Promise.all(
      Object.keys(zip.files).map(async (fileName) => {
        if (fileName.endsWith(".geojson")) {
          const data = await zip.files[fileName].async("string");
          polygonStrings.push(JSON.parse(data));
        }
      })
    );
    return polygonStrings;
  } catch (error) {
    console.error("Error when fetching the polygons: ", error);
    throw error;
  }
}

/**
 * Fetches orthophoto data from the server.
 * @param {string} orthophotoRoute - The route to fetch orthophoto data.
 * @returns {Promise<Object>} The parsed georaster of the orthophoto data from the server.
 * @throws Will throw an error if fetching the orthophoto data fails.
 */
async function fetchOrthophoto(orthophotoRoute) {
  try {
    const response = await fetch(orthophotoRoute);
    const arrayBuffer = await response.arrayBuffer();
    return await parseGeoraster(arrayBuffer);
  } catch (error) {
    console.error("Error fetching Orthophoto data:", error);
    throw error;
  }
}

/**
 * Shows the progress bar and hides the map element.
 * @param {Object} mapElement - The jQuery object for the map element.
 * @param {Object} progressBarContainerElement - The jQuery object for the progress bar container element.
 * @param {Object} sidebarElement - The jQuery object for the sidebar element.
 */
function showProgressBarHideMap(
  mapElement,
  progressBarContainerElement,
  sidebarElement
) {
  mapElement.hide();
  sidebarElement.hide();
  progressBarContainerElement.show();
}

/**
 * Hides the progress bar and shows the map element.
 * @param {Object} mapElement - The jQuery object for the map element.
 * @param {Object} progressBarContainerElement - The jQuery object for the progress bar container element.
 * @param {Object} sidebarElement - The jQuery object for the sidebar element.
 */
function hideProgressBarShowMap(
  mapElement,
  progressBarContainerElement,
  sidebarElement
) {
  progressBarContainerElement.hide();
  sidebarElement.show();
  mapElement.show();
}

/**
 * Sets up and animates the progress bar.
 * @param {Object} progressBarElement - The jQuery object for the progress bar element.
 * @param {Object} progressStatusElement - The jQuery object for the progress status element.
 * @param {number} expectedTimeInSeconds - The expected time in seconds for the progress bar to complete.
 */
function setupProgressBar(
  progressBarElement,
  progressStatusElement,
  expectedTimeInSeconds
) {
  progressBarElement.animate(
    { width: "100%" },
    {
      duration: expectedTimeInSeconds * 1000,
      easing: "linear",
      step: function (now) {
        let percentage = Math.ceil(now);
        progressBarElement
          .text(percentage + "%")
          .css("width", percentage + "%");
      },
      complete: function () {
        progressBarElement.text("100%");
        progressStatusElement.text("Completed");
      },
    }
  );
}

/**
 * Adds OpenStreetMap basemap to the Leaflet map.
 * @param {Object} map - The Leaflet map object.
 */
function addOSMBasemapToMap(map) {
  L.tileLayer("http://{s}.tile.osm.org/{z}/{x}/{y}.png", {
    attribution:
      '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
}

/**
 * Adds a responsive sidebar to the Leaflet map and opens the home tab of the sidebar.
 * @param {Object} map - The Leaflet map object.
 * @param {string} sidebarDivName - The name of the sidebar div.
 * @param {string} sidebarHomeHrefName - The name of the href for the sidebar home. Used to open it on loading.
 */
function addSidebarToMap(map, sidebarDivName, sidebarHomeHrefName) {
  const sidebar = L.control.sidebar(sidebarDivName).addTo(map);
  sidebar.open(sidebarHomeHrefName);
}

/**
 * Adds polygons to the Leaflet map.
 * @param {Object} map - The Leaflet map object.
 * @param {Array} polygonData - An array of polygon GeoJSON data.
 * @param {Array} distinctColors - An array of distinct colors for polygons.
 * @param {string} polygonBorderColor - The color for polygon borders.
 * @param {number} polygonBorderOpacity - The opacity for polygon borders.
 * @param {number} polygonFillOpacity - The opacity for polygon fills.
 * @param {string} markerBGColor - The background color of the marker.
 * @param {Object} sidebarListElement - The JQuery object of the volume list for the sidebar.
 */
function addPolygonsToMap(
  map,
  polygonData,
  distinctColors,
  polygonBorderColor,
  polygonBorderOpacity,
  polygonFillOpacity,
  markerBGColor,
  sidebarListElement
) {
  let polygonNumber = 1;
  polygonData.forEach((polygonJSON) => {
    const polygonLayer = createVolumePolygonLayerOnMap(
      map,
      polygonJSON,
      distinctColors,
      polygonBorderColor,
      polygonBorderOpacity,
      polygonFillOpacity
    );

    createVolumePolygonMarkerOnMap(
      map,
      polygonJSON["features"][0]["properties"]["centroid"],
      markerBGColor,
      polygonNumber,
      polygonJSON["features"][0]["properties"]["volume_above_m3"],
      polygonJSON["features"][0]["properties"]["volume_below_m3"],
      polygonJSON["features"][0]["properties"]["volume_above_t"],
      polygonJSON["features"][0]["properties"]["volume_below_t"]
    );

    addPolygonToSidebar(
      map,
      sidebarListElement,
      polygonNumber,
      polygonLayer,
      polygonJSON["features"][0]["properties"]["volume_above_t"],
      polygonJSON["features"][0]["properties"]["volume_below_t"]
    );

    polygonNumber += 1;
  });
}

/**
 * Creates and adds a polygon layer to the map.
 * @param {Object} map - The Leaflet map object.
 * @param {Object} polygonJSON - The GeoJSON data for the polygon.
 * @param {Array} distinctColors - An array of distinct colors for polygons.
 * @param {string} polygonBorderColor - The color for polygon borders.
 * @param {number} polygonBorderOpacity - The opacity for polygon borders.
 * @param {number} polygonFillOpacity - The opacity for polygon fills.
 * @returns {Object} polygonLayer - The layer of the polygon.
 */
function createVolumePolygonLayerOnMap(
  map,
  polygonJSON,
  distinctColors,
  polygonBorderColor,
  polygonBorderOpacity,
  polygonFillOpacity
) {
  const polygonFillColor =
    distinctColors[Math.floor(Math.random() * distinctColors.length)];

  const polygonLayer = L.geoJSON(polygonJSON, {
    style: {
      color: polygonBorderColor,
      opacity: polygonBorderOpacity,
      fillColor: polygonFillColor,
      fillOpacity: polygonFillOpacity,
    },
  }).addTo(map);

  return polygonLayer;
}

/**
 * Creates and adds a marker for the polygon centroid to the map.
 * @param {Object} map - The Leaflet map object.
 * @param {Array} centroidCoordinatesLongLat - The coordinates of the centroid in [longitude, latitude] format.
 * @param {string} markerBGColor - The background color of the marker.
 * @param {number} markerNumber - The number of the marker.
 * @param {number} volumeAbovem3 - The volume above the base height in cubic meters.
 * @param {number} volumeBelowm3 - The volume below the base height in cubic meters.
 * @param {number} volumeAboveT - The volume above the base height in tons.
 * @param {number} volumeBelowT - The volume below the base height in tons.
 */
function createVolumePolygonMarkerOnMap(
  map,
  centroidCoordinatesLongLat,
  markerBGColor,
  markerNumber,
  volumeAbovem3,
  volumeBelowm3,
  volumeAboveT,
  volumeBelowT
) {
  centroidCoordinatesLongLat.reverse();

  const marker = L.marker(centroidCoordinatesLongLat, {
    icon: new L.AwesomeNumberMarkers({
      number: markerNumber,
      markerColor: markerBGColor,
    }),
  }).addTo(map);

  marker.bindPopup(
    createVolumeMarkerPopupString(
      volumeAbovem3,
      volumeBelowm3,
      volumeAboveT,
      volumeBelowT
    )
  );

  marker.on("click", (e) => e.target.getPopup());
}

/**
 * Creates a popup string for the volume marker inside the centroid of the polygons.
 * @param {number} volumeAbovem3 - The volume above the base height in cubic meters.
 * @param {number} volumeBelowm3 - The volume below the base height in cubic meters.
 * @param {number} volumeAboveT - The volume above the base height in tons.
 * @param {number} volumeBelowT - The volume below the base height in tons.
 * @returns {string} - The formatted popup string.
 */
function createVolumeMarkerPopupString(
  volumeAbovem3,
  volumeBelowm3,
  volumeAboveT,
  volumeBelowT
) {
  const popupString =
    `<div>Volume above Base Height (m³):&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${volumeAbovem3.toFixed(
      2
    )}</div>` +
    `<div>Volume above Base Height (t):&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${volumeAboveT.toFixed(
      2
    )}</div>` +
    `<div>Volume below Base Height (m³):&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${volumeBelowm3.toFixed(
      2
    )}</div>` +
    `<div>Volume below Base Height (t):&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${volumeBelowT.toFixed(
      2
    )}</div>`;
  return popupString;
}

/**
 *
 * @param {Object} map - The Leaflet map object.
 * @param {Object} sidebarListElement - The JQuery object of the volume list for the sidebar.
 * @param {number} polygonNumber - The number of the polygon.
 * @param {Object} polygonLayer  - The layer of the polygon.
 * @param {number} volumeAboveT - The volume of the polygon above the base height in tons.
 * @param {number} volumeBelowT - The volume of the polygon above the below height in tons.
 */
function addPolygonToSidebar(
  map,
  sidebarListElement,
  polygonNumber,
  polygonLayer,
  volumeAboveT,
  volumeBelowT
) {
  const listItem = document.createElement("li");

  const numberCircle = document.createElement("div");
  numberCircle.className = "number-circle";
  numberCircle.textContent = polygonNumber;

  const dataContainer = document.createElement("div");
  dataContainer.className = "data";

  const volumeAbove = document.createElement("div");
  volumeAbove.textContent = `Volumen (über in t): ${Math.round(volumeAboveT)}`;

  const volumeBelow = document.createElement("div");
  volumeBelow.textContent = `Volumen (unter in t): ${Math.round(volumeBelowT)}`;

  dataContainer.appendChild(volumeAbove);
  dataContainer.appendChild(volumeBelow);

  listItem.appendChild(numberCircle);
  listItem.appendChild(dataContainer);
  listItem.addEventListener("click", function () {
    map.fitBounds(polygonLayer.getBounds());
    const originalStyle = polygonLayer.options.style;
    polygonLayer.setStyle({
      fillColor: "red",
      fillOpacity: "red",
    });
    setTimeout(() => {
      polygonLayer.setStyle(originalStyle);
    }, 1000);
  });

  sidebarListElement.append(listItem);
}

/**
 * Creates an adds an orthophoto layer to the map.
 * @param {Object} map - The Leaflet map object.
 * @param {Object} orthophotoData - The orthophoto data as georaster.
 * @param {number} mapOpacity - The opacity for the orthophoto layer.
 * @param {number} mapResolution - The resolution for the orthophoto layer.
 * @returns {Object} - The orthophoto layer.
 */
function addOrthophotoToMap(map, orthophotoData, mapOpacity, mapResolution) {
  const layer = new GeoRasterLayer({
    georaster: orthophotoData,
    opacity: mapOpacity,
    resolution: mapResolution,
  });
  layer.addTo(map);

  return layer;
}

// Initialise the page when the document is ready
$(document).ready(function () {
  initializePage(
    progressBarElement,
    progressStatusElement,
    expectedTimeInSeconds,
    mapElement,
    progressBarContainerElement,
    volumeCalculationRoute,
    polygonRoute,
    orthophotoRoute,
    distinctColors,
    polygonBorderColor,
    polygonBorderOpacity,
    polygonFillOpacity,
    markerBGColor,
    mapOpacity,
    mapResolution,
    sidebarDivName,
    sidebarHomeHrefName,
    sidebarElement,
    sidebarListElement
  );
});
