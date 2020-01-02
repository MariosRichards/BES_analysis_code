function gridData(granularity) {
    var data = new Array();
    var xpos = 1; //starting xpos and ypos at 1 so the stroke will show when we make the grid below
    var ypos = 1;
   
for(var cell = 0; cell < granularity * granularity; cell++) {
        data.push( {
            id: cell,
            color: '#ffffff',
            xpos: cell % granularity,
            ypos: Math.floor( cell / granularity),
            support: 0,
        } );
}
    return data;
}

var window_x = 700;
var window_y = 700;

var granularity = 50;
var max_distance = (granularity/2)**2
//granularity ** 4;
var width = window_x/granularity;
var height = window_y/granularity;
var turn_delay = .001;

var gridData = gridData(granularity);
var partyData = new Array();
partyData.push( { xpos: 7, ypos: 12, color:'#597cf4', support:0, last_support:0, dx:0, dy:0  } );
partyData.push( { xpos: 33, ypos: 46, color:'#f45962', support:0, last_support:0, dx:0, dy:0  } );
// red = #f45962 , blue =#597cf4, white=#ffffff
var grid = d3.select("#grid")
    .append("svg")
    .attr("width",window_x.toString().concat("px"))
    .attr("height",window_y.toString().concat("px"))
    .on('click', function() {
        var coords = d3.mouse(this);
        partyData.push( { xpos: Math.floor(coords[0]*granularity/window_x), ypos: Math.floor(coords[1]*granularity/window_y), color:'#'+Math.floor(Math.random()*16777215).toString(16), support:0, last_support:0, dx:0,dy:0  } );
        // drawCircle(coords[0], coords[1], getRandom(5,50));
    });

// Create data = list of groups
var allGroup = [".001", "1", "100", "1500"]
    
// Initialize the button
var dropdownButton = d3.select("#grid")
  .append('select')    
    
// add the options to the button
dropdownButton // Add a button
  .selectAll('myOptions') // Next 4 lines add 6 options = 6 colors
 	.data(allGroup)
  .enter()
	.append('option')
  .text(function (d) { return d; }) // text showed in the menu
  .attr("value", function (d) { return d; }) // corresponding value returned by the button

// A function that update the color of the circle
function updateChart(mycolor) {
  turn_delay = parseFloat(mycolor);
  // zeCircle
    // .transition()
    // .duration(1000)
    // .style("fill", mycolor)
}

// When the button is changed, run the updateChart function
dropdownButton.on("change", function(d) {

    // recover the option that has been chosen
    var selectedOption = d3.select(this).property("value")

    // run the updateChart function with this selected option
    updateChart(selectedOption)
})

    
    
function support_change(gridData, partyData, party_no, dx, dy){
  var party = partyData[party_no];
  party.xpos = party.xpos + dx;
  party.ypos = party.ypos + dy;
  
  if ( party.xpos<0 || party.xpos>=granularity ) { 
    party.xpos = party.xpos - dx;
    party.ypos = party.ypos - dy;    
    return -1
    }
  if ( party.ypos<0 || party.ypos>=granularity ) {
    party.xpos = party.xpos - dx;
    party.ypos = party.ypos - dy;        
    return -1
    }
    
  for(var j = 0; j < partyData.length;j++){
    var party = partyData[j];
  //  party.last_support = party.support;
    party.support = 0;
    if ( (j!=party_no) && (party.xpos == partyData[party_no].xpos) && (party.ypos == partyData[party_no].ypos) )
    {
        party.xpos = party.xpos - dx;
        party.ypos = party.ypos - dy;    
        return -1
    }
  }
      
  for(var i = 0; i < gridData.length;i++){
    var cell = gridData[i];
    var distance = max_distance;
    for(var j = 0; j < partyData.length;j++){
        var party = partyData[j];
        var party_cell_dist = (party.xpos - cell.xpos)*(party.xpos - cell.xpos) + (party.ypos - cell.ypos)*(party.ypos - cell.ypos) ;
        if (party_cell_dist<distance) {
            distance = party_cell_dist;
           // cell.color = party.color;
            cell.support = j;
        }
    }  
    var party = partyData[cell.support];
    party.support = party.support+1;
    // cell.color = '#'+Math.floor(Math.random()*16777215).toString(16);
  }
  var party = partyData[party_no];
  party.xpos = party.xpos - dx;
  party.ypos = party.ypos - dy;  
  return party.support
}




function choose_move( move )
{
    var dx,dy;
    if (move==0) {dx = 0, dy = 1}
    if (move==1) {dx = 1, dy = 0}
    if (move==2) {dx = 0, dy = -1}
    if (move==3) {dx = -1, dy = 0}
    return [dx,dy]
}
    
var party_turn=0;

function update(){
  // move parties  
  // move only one party per update
  // for(var j = 0; j < partyData.length;j++){
    j = party_turn;
    party_turn = (party_turn+1)%partyData.length;
    var party = partyData[j];
    
        // var acceptableMove = false;
        // while (! acceptableMove)
        // {
            // acceptableMove=true;
            // var move = Math.floor(Math.random()*4) // 0,1,2,3 compass directions
            // if (move==0) {party.dx = 0, party.dy = 1}
            // if (move==1) {party.dx = 1, party.dy = 0}
            // if (move==2) {party.dx = 0, party.dy = -1}
            // if (move==3) {party.dx = -1, party.dy = 0}
            // if ( (party.xpos+party.dx)<=0 || (party.xpos+party.dx)>=granularity ) { acceptableMove=false }
            // if ( (party.ypos+party.dy)<=0 || (party.ypos+party.dy)>=granularity ) { acceptableMove=false }
        // }
        // party.xpos = party.xpos + party.dx
        // party.ypos = party.ypos + party.dy
        
    var best_move = -1;
    var best_support = party.support;
    for(var move = 0; move <4; move++){
        
        const [dx, dy] = choose_move(move);
        // console.log(dx,dy)
        var support = support_change(gridData, partyData, j, dx, dy)
        if (support > best_support)
        {
            best_support = support;
            best_move = move;
        }
        
    }
    if (best_move >-1)
    {
        const [dx, dy] = choose_move(best_move);
        // var party = partyData[party_no];
        party.xpos = party.xpos + dx;
        party.ypos = party.ypos + dy;        
    }
        
        
        // if (Math.random()>.5) { party.xpos = party.xpos+1 }
        // else { party.xpos = party.xpos-1 }
        // if (Math.random()>.5) { party.ypos = party.ypos+1 }
        // else { party.ypos = party.ypos-1 }
        // party.xpos = (granularity+party.xpos)%granularity
        // party.ypos = (granularity+party.ypos)%granularity
        
  // }
    
  for(var j = 0; j < partyData.length;j++){
    var party = partyData[j];
 //   party.last_support = party.support;
    party.support = 0;
  }
      
      
  // final update!
  for(var i = 0; i < gridData.length;i++){
    var cell = gridData[i];
    var distance = max_distance;
    cell.color = '#828282'; //default non-voting
    for(var j = 0; j < partyData.length;j++){
        var party = partyData[j];
        var party_cell_dist = (party.xpos - cell.xpos)*(party.xpos - cell.xpos) + (party.ypos - cell.ypos)*(party.ypos - cell.ypos) ;
        if (party_cell_dist<distance) {
            distance = party_cell_dist;
            cell.color = party.color;
            cell.support = j;
        }
    }  
    var party = partyData[cell.support];
    party.support = party.support+1;
  }  
  

var cell = grid.selectAll(".cell")
    .data(gridData)

    cell.enter().append("rect")
    .attr("class","cell")
    .attr("x", function(d) { return d.xpos * width; })
    .attr("y", function(d) { return d.ypos * height; })
    .attr("width", width )
    .attr("height", height)
    .style("fill", function(d){ return d.color; })
    // .style("stroke", "#222");

    cell.transition()
			    .duration(turn_delay)
			    .delay(turn_delay)
			    .style('fill', 
                function(d){ 
                    return d.color;
                     });

var party = grid.selectAll(".party")
    .data(partyData)
    
    party.enter().append("circle")
    .attr("class","party")
    .attr("cx", function(d) { return d.xpos * width; })
    .attr("cy", function(d) { return d.ypos * height; })
    .attr("r", width )
    // .attr("width", width )
    // .attr("height", height)
    // .style("fill", function(d){ return d.color; })    
    // .style("fill", "#222") 
    .style("fill", "#ffffff") 
    .style("stroke", "#222")


    
  // circle
      // .attr("cx", function(d) { d.x += d.dx; if (d.x > w) d.x -= w; else if (d.x < 0) d.x += w; return d.x; })
      // .attr("cy", function(d) { d.y += d.dy; if (d.y > h) d.y -= h; else if (d.y < 0) d.y += h; return d.y; });    
  
    // svg.on('click', function() {
        // var coords = d3.mouse(this);
        // console.log(coords);
        // drawCircle(coords[0], coords[1], getRandom(5,50));
    // });  
    
    
    party.transition()
			    .duration(turn_delay)
			    .delay(turn_delay)
                .attr("cx", function(d) { return d.xpos * width; })
                .attr("cy", function(d) { return d.ypos * height; })                
			    .style('fill', 
                function(d){ 
                    return d.color;
                     })
                // .attr("dx", function(d){return -20})
                // .append("text")
                // .text(function(d){return d.support.toString()});
                // .attr({
                  // "text-anchor": "middle",
                  // "font-size": function(d) {
                    // return d.r / ((d.r * 10) / 100);
                  // },
                  // "dy": function(d) {
                    // return d.r / ((d.r * 25) / 100);
                  // }
                // });
                .selectAll('circle')
    party.enter()    
        .append("text")
        .text(function(d){return d.support.toString()});
}

setInterval(update, turn_delay   );







