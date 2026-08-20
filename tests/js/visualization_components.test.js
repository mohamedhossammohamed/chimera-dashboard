// tests/js/visualization_components.test.js
import test from 'node:test';
import assert from 'node:assert';
import { setupMockDOM, MockElement } from '../../test/helpers/mock_dom.js';

// Initialise the shared full-featured mock DOM before any test executes.
setupMockDOM();

// The shared mock_dom.js does not expose createTextNode or document.body,
// which the SVG renderers rely on. Augment the global document with the
// minimal additions required by standard_components.js.
if (!globalThis.document.createTextNode) {
  globalThis.document.createTextNode = function (text) {
    return { nodeType: 3, textContent: String(text), nodeName: '#text' };
  };
}
if (!globalThis.document.body) {
  globalThis.document.body = new MockElement('body');
}

import {
  renderClevelandBulletStrip,
  renderKaplanMeierSVG,
  renderEAUScorecard,
  renderConcordanceMatrix
} from '../../dashboard/js/standard_components.js';

test('renderClevelandBulletStrip renders SVG bullet strip with threshold bands', () => {
  const container = document.createElement('div');
  const thresholds = {
    domain: [0, 50],
    normal_max: 4,
    borderline_max: 10,
    unit: 'ng/mL'
  };

  renderClevelandBulletStrip(container, 15.2, thresholds);
  assert.strictEqual(container.children.length, 1);
  const svg = container.children[0];
  assert.strictEqual(svg.tagName.toLowerCase(), 'svg');
  assert.strictEqual(svg.getAttribute('viewBox'), '0 0 400 70');
});

test('renderClevelandBulletStrip handles missing values gracefully', () => {
  const container = document.createElement('div');
  const thresholds = { domain: [0, 50], normal_max: 4, borderline_max: 10, unit: 'ng/mL' };

  renderClevelandBulletStrip(container, null, thresholds);
  assert.strictEqual(container.children.length, 1);
  const svg = container.children[0];
  // Verify MISSING text was appended
  const textEl = svg.children.find(c => c.textContent === 'MISSING');
  assert.ok(textEl, 'Should render MISSING badge');
});

test('renderClevelandBulletStrip handles equal domain endpoints without division by zero', () => {
  const container = document.createElement('div');
  const thresholds = { domain: [5, 5], normal_max: 5, borderline_max: 5, unit: 'ng/mL' };

  // Should not throw or generate NaN/Infinity attributes
  renderClevelandBulletStrip(container, 5.0, thresholds);
  assert.strictEqual(container.children.length, 1);
});

test('renderKaplanMeierSVG renders staircase step function and markers', () => {
  const container = document.createElement('div');
  const timePoints = [0, 12, 24, 36, 48, 60];
  const survivalProbs = [1.0, 0.95, 0.85, 0.70, 0.60, 0.50];
  const eventStatus = [0, 0, 1, 0, 1, 0];

  renderKaplanMeierSVG(container, timePoints, survivalProbs, eventStatus, 36);
  assert.strictEqual(container.children.length, 1);
  const wrapper = container.children[0];
  assert.strictEqual(wrapper.children.length, 3); // svg + risk table + download button

  const svg = wrapper.children[0];
  const path = svg.children.find(c => c.tagName.toLowerCase() === 'path');
  assert.ok(path, 'Should contain step function path');
  const pathD = path.getAttribute('d');
  assert.ok(pathD.startsWith('M '), 'Path must start with M');
  assert.ok(pathD.includes(' L '), 'Step function path must use orthogonal L segments');
});

test('renderKaplanMeierSVG handles missing/empty data gracefully', () => {
  const container = document.createElement('div');
  renderKaplanMeierSVG(container, [], [], [], null);
  assert.strictEqual(container.children.length, 1);
  const wrapper = container.children[0];
  const svg = wrapper.children[0];
  const textEl = svg.children.find(c => c.textContent === '[DATA NOT RECORDED]');
  assert.ok(textEl, 'Should render [DATA NOT RECORDED]');
});

test('renderEAUScorecard renders 5 tiers with active selection', () => {
  const container = document.createElement('div');
  renderEAUScorecard(container, 'High', 'PSA >= 10');

  // Should have grid and criteria line (style tag removed, now inline styles)
  assert.strictEqual(container.children.length, 2);
  const grid = container.children[0]; // grid is now first child (no style tag)
  assert.strictEqual(grid.children.length, 5); // 5 tier cards

  // The High card should be active (inline styles, no className — check ACTIVE badge)
  const highCard = grid.children[3];
  assert.ok(highCard.children.some(c => c.textContent === 'ACTIVE'), 'High card should have ACTIVE badge');
});

test('renderConcordanceMatrix renders 5x5 PI-RADS vs ISUP grid', () => {
  const container = document.createElement('div');
  renderConcordanceMatrix(container, 4, 3);

  assert.strictEqual(container.children.length, 1);
  const wrap = container.children[0];
  // Verify table is inside wrap
  const table = wrap.children.find(c => c.tagName.toLowerCase() === 'table');
  assert.ok(table, 'Should render table');
  const tbody = table.children.find(c => c.tagName.toLowerCase() === 'tbody');
  assert.ok(tbody, 'Should contain tbody');
  assert.strictEqual(tbody.children.length, 5); // 5 rows for ISUP 1-5
});
