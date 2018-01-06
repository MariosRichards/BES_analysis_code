var w = window,
    d = document,
    e = d.documentElement,
    g = d.getElementsByTagName('body')[0],
    width = w.innerWidth || e.clientWidth || g.clientWidth,
    height = w.innerHeight|| e.clientHeight|| g.clientHeight,
    margin = 12,
    width = width - margin*2,
    height = height - margin*2;

var int_width = 1000
var int_height = 1000

d3.json("force.json", function(json) {
  var force = d3.layout.force()
      .nodes(json.nodes)
      .links(json.links)
      .charge(-130)
      .linkDistance(function(link) {
       return 10+(.1/((link.weight/10)**2));
    })
      .size([int_width, int_height])
      .on("tick", tick) // new
      .start();

      // .linkDistance(function(link) {
       // return 500/link.weight;
    // })
  
var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", "0 0 1200 1000")
    .attr("preserveAspectRatio", "none");
    //.append("g");
        // .call(d3.behavior.zoom().scaleExtent([1, 8]).on("zoom", zoom))
    // .append("g");


// 2024, 954

function updateWindow(){
    width = w.innerWidth || e.clientWidth || g.clientWidth;
    height = w.innerHeight|| e.clientHeight|| g.clientHeight;
    width = width - margin*2,
    height = height - margin*2;  
    svg.attr("width", width).attr("height", height);
    force.size([int_width, int_height]).resume();
}
window.onresize = updateWindow;

// build the arrow.
svg.append("svg:defs").selectAll("marker")
    .data(["end"])      // Different link/path types can be defined here
  .enter().append("svg:marker")    // This section adds in the arrows
    .attr("id", String)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 15)
    .attr("refY", -1.5)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
  .append("svg:path")
    .attr("d", "M0,-5L10,0L0,5");

// add the links and the arrows
var path = svg.append("svg:g").selectAll("path")
    .data(force.links())
  .enter().append("svg:path")
    .attr("class", function(d) { return "link " + d.type; })
    .attr("class", "link")
    .attr("marker-end", "url(#end)")
    .style("stroke-width", function(d) {return d.weight*20});

// define the nodes
var node = svg.selectAll(".node")
    .data(force.nodes())
  .enter().append("g")
    .attr("class", "node")
    .call(force.drag);

// add the nodes
node.append("circle")
    .attr("r", 5);

// add the text 
node.append("text")
    .attr("x", 12)
    .attr("dy", ".35em")
    .text(function(d) { return d.name; });

function zoom() {
  svg.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
}

// add the curvy lines
function tick() {
    path.attr("d", function(d) {
        var dx = d.target.x - d.source.x,
            dy = d.target.y - d.source.y,
            dr = Math.sqrt(dx * dx + dy * dy);
        return "M" + 
            d.source.x + "," + 
            d.source.y + "A" + 
            dr + "," + dr + " 0 0,1 " + 
            d.target.x + "," + 
            d.target.y;
    });
    // .style("stroke-width", function(d) {return (2**d.weight)/3000});

    node
        .attr("transform", function(d) { 
  	    return "translate(" + d.x + "," + d.y + ")"; });
}
});

