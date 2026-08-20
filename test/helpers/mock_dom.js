// test/helpers/mock_dom.js
// Lightweight Mock DOM for headless Node.js testing of SVG/DOM Renderers (B1-B5)

export class MockElement {
  constructor(tagName, ns = null) {
    this.tagName = tagName.toUpperCase();
    this.nodeName = this.tagName;
    this.namespaceURI = ns;
    this.attributes = {};
    this.children = [];
    this.childNodes = this.children;
    this.style = {};
    this._innerHTML = '';
    this._textContent = '';
    this.className = '';
    this.parentNode = null;
  }

  addEventListener(event, handler) {
    if (!this._listeners) this._listeners = {};
    if (!this._listeners[event]) this._listeners[event] = [];
    this._listeners[event].push(handler);
  }

  removeEventListener(event, handler) {
    if (!this._listeners || !this._listeners[event]) return;
    const idx = this._listeners[event].indexOf(handler);
    if (idx !== -1) this._listeners[event].splice(idx, 1);
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === 'id') this.id = String(value);
    if (name === 'class') this.className = String(value);
  }

  getAttribute(name) {
    return this.attributes[name] ?? null;
  }

  hasAttribute(name) {
    return name in this.attributes;
  }

  removeAttribute(name) {
    delete this.attributes[name];
    if (name === 'id') delete this.id;
    if (name === 'class') this.className = '';
  }

  appendChild(child) {
    if (child) {
      child.parentNode = this;
      this.children.push(child);
    }
    return child;
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx !== -1) {
      this.children.splice(idx, 1);
      child.parentNode = null;
    }
    return child;
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(val) {
    this._innerHTML = String(val);
    this.children = [];
    this.childNodes = [];
    if (this._innerHTML.includes('class="')) {
      const matches = this._innerHTML.matchAll(/class="([^"]+)"/g);
      for (const m of matches) {
        const dummy = new MockElement('div');
        dummy.className = m[1];
        dummy.setAttribute('class', m[1]);
        this.appendChild(dummy);
      }
    }
  }

  get textContent() {
    return this._textContent;
  }

  set textContent(val) {
    this._textContent = String(val);
  }

  querySelector(selector) {
    const all = this.querySelectorAll(selector);
    return all.length > 0 ? all[0] : null;
  }

  querySelectorAll(selector) {
    const res = [];
    const match = el => {
      if (selector.startsWith('#') && (el.id === selector.slice(1) || el.attributes['id'] === selector.slice(1))) return true;
      if (selector.startsWith('.') && (el.className.split(' ').includes(selector.slice(1)) || el.attributes['class'] === selector.slice(1))) return true;
      if (el.tagName.toLowerCase() === selector.toLowerCase()) return true;
      return false;
    };

    const traverse = node => {
      for (const ch of node.children) {
        if (ch instanceof MockElement) {
          if (match(ch)) res.push(ch);
          traverse(ch);
        }
      }
    };
    traverse(this);
    return res;
  }

  getContext(type) {
    if (type === '2d') {
      return {
        font: '',
        measureText: (text) => ({ width: (text ? String(text).length : 0) * 7.5 }),
      };
    }
    return null;
  }
}

export function setupMockDOM() {
  const elementsById = new Map();

  globalThis.document = {
    createElement(tagName) {
      return new MockElement(tagName);
    },
    createElementNS(ns, tagName) {
      return new MockElement(tagName, ns);
    },
    getElementById(id) {
      if (!elementsById.has(id)) {
        const el = new MockElement('div');
        el.setAttribute('id', id);
        elementsById.set(id, el);
      }
      return elementsById.get(id);
    },
    reset() {
      elementsById.clear();
    }
  };

  globalThis.window = globalThis;
}
