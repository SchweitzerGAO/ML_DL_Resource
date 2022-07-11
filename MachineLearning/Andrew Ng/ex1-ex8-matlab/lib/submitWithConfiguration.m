function submitWithConfiguration(conf)
%SUBMITWITHCONFIGURATION   Submit assignment for grading
%   submitWithConfiguration(CONF) submits the assignment defined in struct
%   CONF to Coursera for on demand grading of programming assignments. CONF
%   is a configuration struct constructed by one of the submit.m scripts in
%   the assignment folders, e.g. ex1/submit.m
%
%   This function is called internally by the submit scripts and should not
%   be called directly by students
%
%   Latest version: Updated by Brian Buechel in November 2021.
%

parts = Parts(conf);

fprintf('== Submitting solutions | %s...\n', conf.itemName);

% Get email and token from student
tokenFile = 'token.mat';
if exist(tokenFile, 'file')
    load(tokenFile,'token','email');
    [email,token] = promptToken(email, token, tokenFile);
else
    [email,token] = promptToken('', '', tokenFile);
end
if isempty(token)
    fprintf('!! Submission canceled due to empty token. Please try again.\n');
    return
end

% Submit assignment
response = submitParts(conf, email, token, parts);

if isfield(response, 'errorCode')
    % Construct error message from response
    msg = sprintf('!! Submission failed: %s\n', response.errorCode);
    if isfield(response,"message")
        msg = [msg sprintf('!! %s\n',response.message)];
    end
    if isfield(response,'details')
        if isfield(response.details,'learnerMessage')
            msg = [msg sprintf('!! %s\n',response.details.learnerMessage)];
        end
    end
    fprintf(msg);
elseif ~isempty(response)
    % Submission successful! Show results to students
    showFeedback(parts, response);
    save(tokenFile, 'email', 'token');    
end

end
%% Helper functions beyond this point
%% promptToken
function [email,token] = promptToken(email, existingToken, tokenFile)
if (~isempty(email) && ~isempty(existingToken))
    prompt = sprintf( ...
        'Use token from last successful submission (%s)? (Y/n): ', ...
        email);
    reenter = input(prompt, 's');

    if (isempty(reenter) || reenter(1) == 'Y' || reenter(1) == 'y')
        token = existingToken;
        return;
    else
        delete(tokenFile);
    end
end
email = input('Login (email address): ', 's');
token = input('Token: ', 's');
end

%% submitParts
function response = submitParts(conf, email, token, parts)

% Prepare submission
response = '';
submissionUrl = SubmissionUrl(); % Updated
try
    body = makePostBody(conf, email, token, parts);
catch ME
    fprintf('!! Failed to prepare submission: Error in "%s" (line %d)\n!! %s\n', ...
        ME.stack(1).name,ME.stack(1).line,ME.message);
    return
end

% Submit assignment
try
    responseBody = getResponse(submissionUrl, body);
    response = loadjson(responseBody);
catch ME
    fprintf('!! Submission failed: Error in "%s" (line %d)\n!! %s\n', ...
        ME.stack(1).name,ME.stack(1).line,ME.message);
    fprintf('!! Please try again later.\n');
    return
end

end

%% getResponse
function response = getResponse(url, body)
% NEW CURL SUBMISSION FOR WINDOWS AND MAC
if ispc
    % Regex line will escape double quoted objects to format properly for windows libcurl
    % json_command also has -s option to not print out the progress bar
    new_body = regexprep (body, '\"', '\\"'); 
    json_command = sprintf('curl -X POST -s -H "Cache-Control: no-cache" -H "Content-Type: application/json" -d "%s" --ssl-no-revoke "%s"', new_body, url);
else
    json_command = sprintf('curl -X POST -H "Cache-Control: no-cache" -H "Content-Type: application/json" -d '' %s '' --ssl-no-revoke ''%s''', body, url);
end

% Run system command
[code, response] = system(json_command);

% test the success code
if (code~=0)
    msg = sprintf('Submission with curl was not successful (code = %d)',code);

    % Add on the reason for curl failure for common error codes
    simpleLookup = {'Unsupported protocol','Failed to initialize','URL malformed', ...
        '','Couldn''t resolve proxy','Couldn''t resolve host','Failed to connect to host'};
    if any(code == 1:length(simpleLookup))
        msg = [msg ': ' simpleLookup{code}];
    end

    % Throw the error
    error(msg);
end
end

%% makePostBody
function body = makePostBody(conf, email, token, parts)
bodyStruct.assignmentKey = conf.assignmentKey;
bodyStruct.submitterEmail = email;
bodyStruct.secret = token;
bodyStruct.parts = makePartsStruct(conf, parts);

opt.Compact = 1;
body = savejson('', bodyStruct, opt);
end

%% makePartsStruct
function partsStruct = makePartsStruct(conf, parts)
for part = parts
    partId = part{:}.id;
    fieldName = makeValidFieldName(partId);
    outputStruct.output = conf.output(partId);
    partsStruct.(fieldName) = outputStruct;
end
end

%% Parts
function [parts] = Parts(conf) % Updated
parts = {};
for partArray = conf.partArrays
    part.id = partArray{:}{1};
    part.sourceFiles = partArray{:}{2};
    part.name = partArray{:}{3};
    parts{end + 1} = part;
end
end

%% showFeedback
function showFeedback(parts, response)
fprintf('== \n');
fprintf('== %43s | %9s | %-s\n', 'Part Name', 'Score', 'Feedback');
fprintf('== %43s | %9s | %-s\n', '---------', '-----', '--------');
evaluation = response.linked.onDemandProgrammingScriptEvaluations_0x2E_v1{1}(1);
for part = parts
    % NEW PARSING REPONSE BODY
    partEvaluation = evaluation.parts.(makeValidFieldName(part{:}.id));
    partFeedback = partEvaluation.feedback;
    score = sprintf('%d / %3d', partEvaluation.score, partEvaluation.maxScore);
    fprintf('== %43s | %9s | %-s\n', part{:}.name, score, partFeedback);
end
totalScore = sprintf('%d / %d', evaluation.score, evaluation.maxScore);
fprintf('==                                   --------------------------------\n');
fprintf('== %43s | %9s | %-s\n', '', totalScore, '');
fprintf('== \n');
end

%% SubmissionURL
function submissionUrl = SubmissionUrl() % Updated
submissionUrl = 'https://www.coursera.org/api/onDemandProgrammingScriptSubmissions.v1?includes=evaluation';
end
